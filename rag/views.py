# rag/views.py

from django.http import JsonResponse
from .models import Tenant, Document, ChatHistory, DocumentAlert, MultiFileChatSession, DocumentAccess, ExternalDocument
import json
from rest_framework.decorators import api_view
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from knox.models import AuthToken
from rest_framework import generics
from .serializers import *
from django.contrib.auth import authenticate, get_user_model
from django.shortcuts import get_object_or_404
from rest_framework.parsers import MultiPartParser, FormParser
from .utils import (
    extract_text_from_file,
    insert_document_to_vectorstore,
    ask_question,
    retrieve_documents_by_vector_id,
    delete_documents_by_vector_id,
    retrieve_documents_by_vector_ids,
    summarize_context,
    extract_metadata,
    handle_table_query,
    ask_question_for_single_document,
    nlp,
    llm,
)
import tempfile
from pathlib import Path
import time
import logging
import threading
import uuid
import hashlib
import requests
# logging.basicConfig(filename="app.log",level=logging.INFO)

User = get_user_model()

class ProtectedView(APIView):

    def get(self, request,token):
        if not token:
            return Response({'error': 'Token not provided.'}, status=status.HTTP_400_BAD_REQUEST)
        auth_token = get_object_or_404(AuthToken, token_key=token)
        user = auth_token.user
        return Response({
            'user': UserSerializer(user).data,
            'token': token,
            'expiry': auth_token.expiry
        }, status=status.HTTP_200_OK)

class RegisterAPI(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = RegisterSerializer

class LoginView(generics.CreateAPIView):
    serializer_class = LoginSerializer

    def post(self, request, format=None):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        username = data.get('username')
        password = data.get('password')

        if username is None or password is None:
            return Response({'error': 'Please provide both username and password.'},
                            status=status.HTTP_400_BAD_REQUEST)

        user = authenticate(request, username=username, password=password)

        if not user:
            return Response({'error': 'Invalid credentials.'},
                            status=status.HTTP_401_UNAUTHORIZED)

        # Generate token
        token_instance, token = AuthToken.objects.create(user)

        # Serialize user data
        user_data = UserSerializer(user).data

        return Response({
            'token': token_instance.token_key,
            'expiry': token_instance.expiry,
            'user': user_data
        }, status=status.HTTP_200_OK)

class LogoutView(APIView):
    def post(self, request, token=None, format=None):
        if not token:
            return Response({'error': 'Token not provided.'}, status=status.HTTP_400_BAD_REQUEST)

        auth_token = get_object_or_404(AuthToken, token_key=token)
        auth_token.delete()
        return Response({'message': 'Logged out successfully.'}, status=status.HTTP_204_NO_CONTENT)
    

class IngestAPIView(generics.CreateAPIView):
    serializer_class = IngestDocumentSerializer
    queryset = Document.objects.all()  # or a filtered queryset as needed
    
    def post(self, request, token):
        auth_token = get_object_or_404(AuthToken, token_key=token)
        user = auth_token.user
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            # initialize table_answer for downstream
            table_answer = None
            uploaded_file = serializer.validated_data.get('file', None)
            s3_file_url = serializer.validated_data.get('s3_file_url', None)

            source_type = "file"  # or dynamically decide if needed

            if not uploaded_file and not s3_file_url:
                return Response({"error": "No input file provided. Please upload a local file or provide an S3 file URL."}, status=status.HTTP_400_BAD_REQUEST)

            # Decide which source to use
            if uploaded_file:
                file_name = uploaded_file.name
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_name).suffix) as tmp:
                    for chunk in uploaded_file.chunks():
                        tmp.write(chunk)
                    tmp_path = tmp.name
                print(f"[+] Local file uploaded and saved at: {tmp_path}")

            elif s3_file_url:
                from urllib.parse import urlparse
                parsed_url = urlparse(s3_file_url)
                file_name = parsed_url.path.split("/")[-1]
                
                response = requests.get(s3_file_url)
                if response.status_code != 200:
                    return Response({"error": "Failed to download file from S3 URL."}, status=status.HTTP_400_BAD_REQUEST)
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_name).suffix) as tmp:
                    print(f"[+] Downloading file from S3 URL: {s3_file_url}")
                    print(f"[+] Saving file as: {file_name}")
                    tmp.write(response.content)
                    print(f"[+] File downloaded successfully.", tmp)
                    tmp_path = tmp.name

                print(f"[+] File downloaded from S3 and saved at: {tmp_path}")
                
            # Generate a unique vector_id
            vector_id = str(uuid.uuid4())


            try:
                # Extract text and metadata
                tenant = user.tenant
                extracted_text = extract_text_from_file(tmp_path, file_name)
                file_ext = Path(file_name).suffix.lower()
                if not extracted_text.strip():
                    return Response({"error": "No text could be extracted."}, status=status.HTTP_400_BAD_REQUEST)
                # 1) Insert into vector store with progress
                insert_document_to_vectorstore(extracted_text, source_type, file_ext, vector_id)
                # 2) Save metadata in Django and external SQL
                document = Document(tenant=tenant, title=file_name, content=extracted_text, vector_id=vector_id)
                document.save()
                try:
                    ExternalDocument.objects.using('external').create(
                        vector_id=vector_id, title=file_name, content=extracted_text
                    )
                except Exception as e:
                    print(f"[!] External save failed: {e}")
                # 3) Enrich and alerts
                try:
                    self.enrich_document(document, extracted_text, file_ext)
                    self.detect_alerts(document, extracted_text)
                except Exception:
                    pass
                # 4) Clean up
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass
                # 5) Final response with vector_id
                return Response({
                    "message": "File ingested and stored successfully.",
                    "file_name": file_name,
                    "vector_id": vector_id
                }, status=status.HTTP_200_OK)
            except ValueError as ve:
                return Response({"error": str(ve)}, status=status.HTTP_400_BAD_REQUEST)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def get(self, request, token):
        auth_token = get_object_or_404(AuthToken, token_key=token)
        user = auth_token.user
        documents = Document.objects.filter(tenant=user.tenant).defer('vector_id').values().order_by('-uploaded_at')
        if not documents:
            return Response({"message": "No documents found."}, status=status.HTTP_404_NOT_FOUND)
        return Response(DocumentSerializer(documents, many=True).data, status=status.HTTP_200_OK)


    def enrich_document(self,document_obj, file_text, file_ext):
        try:
            if not file_text.strip():
                print("[!] Empty file content, skipping enrichment")
                return

            # Summarize the document (first 3000 chars)
            summary = summarize_context(file_text[:3000])
            # Extract keywords/metadata
            metadata = extract_metadata(file_text, file_ext)

            document_obj.summary = summary
            document_obj.keywords = metadata
            document_obj.save()

            print(f"[+] Enriched document {document_obj.title} with summary and keywords.")

        except Exception as e:
            print(f"[!] Failed to enrich document: {e}")

    def detect_alerts(self, document_obj, file_text):
        try:
            alert_keywords = [
                # Contract & Expiry
                "contract expiry", "contract end date", "renewal deadline", "service termination", "expiry notice",
                # Payments
                "payment due", "payment overdue", "invoice overdue", "late fee", "unpaid invoice", "outstanding balance", "collection notice",
                # Legal Risks
                "breach of contract", "penalty clause", "legal action", "non-compliance", "lawsuit", "settlement",
                # Deadlines
                "submission deadline", "due date", "project deadline", "final notice", "critical timeline",
                # Financial
                "advance payment", "refund request", "debit note", "credit note", "balance payable",
                # Risk Specific
                "termination for cause", "default notice", "breach penalty", "financial exposure",
                # Communication
                "no response received", "pending approval", "awaiting confirmation",
                # Supply Chain
                "shipment delay", "logistics issue", "supply disruption",
                # Tax / Regulatory
                "tax penalty", "compliance audit", "regulatory fine",
                # Partner/Vendor Risks
                "partner dispute", "vendor breach", "service level failure",

                # Additional keywords from the provided document
                "invoice", "payment summary", "total amount", "booking fees", "ride charge",
                "cancellation policy", "cancellation fees", "cancellation notice", "cancellation confirmation",
            ]

            file_text_lower = file_text.lower()

            for keyword in alert_keywords:
                if keyword in file_text_lower:
                    idx = file_text_lower.find(keyword)
                    snippet = file_text[max(0, idx-100): idx+100]  # Extract 100 chars before and after
                    DocumentAlert.objects.create(
                        document=document_obj,
                        keyword=keyword,
                        snippet=snippet
                    )
                    print(f"[+] Alert created for '{keyword}' in document {document_obj.title}")

        except Exception as e:
            print(f"[!] Failed to detect alerts: {e}")


class AskAPIView(generics.CreateAPIView):
    serializer_class = AskQuestionSerializer
    def post(self, request, token):
        auth_token = get_object_or_404(AuthToken, token_key=token)
        user = auth_token.user
        # user = auth_token.user
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            question = serializer.validated_data['question']
            vector_id = serializer.validated_data.get('vector_id')  # Optional field in serializer
            chat_history = serializer.validated_data.get('chat_history')  # Optional field in serializer
            user_identifier = serializer.validated_data['user_identifier']
            source_type = "file"  # or dynamic based on your design

            if not question:
                return Response(
                    {"error": "A question must be provided."},
                    status=status.HTTP_400_BAD_REQUEST
                )
            # If this is a table file, first check structured query, then fallback to DataFrame agent
            # Special handling for single-file table queries
            if vector_id:
                document = get_object_or_404(Document, vector_id=vector_id, tenant=user.tenant)
                if user_identifier != user.email and not DocumentAccess.objects.filter(
                    document=document, user_identifier=user_identifier
                ).exists():
                    return Response({'error': 'No access to this document.'}, status=status.HTTP_403_FORBIDDEN)
                from pathlib import Path
                file_ext = Path(document.title).suffix.lower()
                if file_ext in ['.csv', '.xls', '.xlsx']:
                    # Try DataFrame agent for natural-language table queries (handles multi-criteria)
                    try:
                        from io import StringIO
                        import pandas as pd
                        from langchain_experimental.agents import create_pandas_dataframe_agent
                        from .utils import llm
                        df = pd.read_csv(StringIO(document.content))
                        agent = create_pandas_dataframe_agent(llm, df, verbose=False)
                        df_answer = agent.run(question)
                        # Normalize the answer
                        if isinstance(df_answer, str):
                            tbl_list = [s.strip() for s in df_answer.split(',') if s.strip()]
                        elif isinstance(df_answer, list):
                            tbl_list = df_answer
                        else:
                            tbl_list = [str(df_answer)]
                        return Response({
                            'vector_answer': '',
                            'table_answer': tbl_list,
                            'sql_answer': tbl_list
                        }, status=status.HTTP_200_OK)
                    except Exception as e:
                        print(f"[!] Table agent error: {e}")
                    # Fallback: simple structured table query
                    from .utils import handle_table_query
                    table_res = handle_table_query(document.content, question)
                    if table_res is not None:
                        if isinstance(table_res, str):
                            tbl_list = [s.strip() for s in table_res.split(',') if s.strip()]
                        elif isinstance(table_res, list):
                            tbl_list = table_res
                        else:
                            tbl_list = [str(table_res)]
                        return Response({
                            'vector_answer': '',
                            'table_answer': tbl_list,
                            'sql_answer': tbl_list
                        }, status=status.HTTP_200_OK)
            try:
                # 1) Vector DB retrieval & LLM QA
                if vector_id:
                    documents = retrieve_documents_by_vector_id(vector_id)
                    if not documents:
                        return Response({"error": "No documents found for the given vector_id."}, status=status.HTTP_404_NOT_FOUND)
                    vector_answer = ask_question_for_single_document(
                        question, source_type=source_type,
                        documents=documents, chat_history=chat_history
                    )
                    chat, _ = ChatHistory.objects.get_or_create(
                        vector_id=vector_id, user_identifier=user_identifier,
                        tenant=user.tenant, defaults={'history': []}
                    )
                    chat.history.extend([
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": vector_answer}
                    ])
                    chat.save()
                else:
                    vector_answer = ask_question(question, source_type, chat_history)

                # 2) Table agent across CSV/Excel docs for this tenant
                table_answer = []
                from pathlib import Path
                from .utils import handle_table_query
                for doc in Document.objects.filter(tenant=user.tenant):
                    ext = Path(doc.title).suffix.lower()
                    if ext in ['.csv', '.xls', '.xlsx']:
                        try:
                            ext_doc = ExternalDocument.objects.using('external').get(vector_id=doc.vector_id)
                            res = handle_table_query(ext_doc.content, question)
                            if res is not None:
                                table_answer.append({doc.title: res})
                        except Exception as e:
                            print(f"[!] Table query error for {doc.title}: {e}")
                if not table_answer:
                    table_answer = None

                # 3) SQL substring fallback across all tenant docs
                sql_answer = []
                ql = question.lower()
                for doc in Document.objects.filter(tenant=user.tenant):
                    try:
                        ext_doc = ExternalDocument.objects.using('external').get(vector_id=doc.vector_id)
                        content_lower = ext_doc.content.lower()
                        if ql in content_lower:
                            idx = content_lower.find(ql)
                            snippet = ext_doc.content[max(idx-50,0):min(idx+150, len(ext_doc.content))]
                            sql_answer.append({doc.title: snippet})
                    except Exception:
                        continue
                if not sql_answer:
                    sql_answer = []

                # 4) SpaCy-based ORG fallback for non-table docs when others miss
                if (not vector_answer or vector_answer.lower().startswith('without')) and not table_answer and not sql_answer:
                    orgs = set()
                    from pathlib import Path
                    for doc in Document.objects.filter(tenant=user.tenant):
                        ext = Path(doc.title).suffix.lower()
                        if ext not in ['.csv', '.xls', '.xlsx']:
                            try:
                                ext_doc = ExternalDocument.objects.using('external').get(vector_id=doc.vector_id)
                                doc_nlp = nlp(ext_doc.content)
                                for ent in doc_nlp.ents:
                                    if ent.label_ == 'ORG':
                                        for sent in doc_nlp.sents:
                                            if ent.start_char >= sent.start_char and ent.end_char <= sent.end_char:
                                                if 'sri lanka' in sent.text.lower():
                                                    orgs.add(ent.text)
                            except Exception:
                                continue
                    if orgs:
                        vector_answer = ', '.join(orgs)
                        if not sql_answer:
                            sql_answer = list(orgs)
                        if not table_answer:
                            table_answer = list(orgs)

                # If table and SQL answers are empty but vector has list output, parse it
                if (not table_answer) and vector_answer:
                    import re
                    names = []
                    for line in vector_answer.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        m = re.match(r"^\s*\d+[\).\-]?\s*(.*)", line)
                        if m:
                            names.append(m.group(1).strip())
                        else:
                            names.append(line)
                    if names:
                        table_answer = names
                        if not sql_answer:
                            sql_answer = names
                # Return combined answers
                return Response({
                    "vector_answer": vector_answer,
                    "table_answer": table_answer,
                    "sql_answer": sql_answer
                }, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        

class RetrieveByVectorIdAPIView(APIView):
    def get(self, request, token, vector_id):
        auth_token = get_object_or_404(AuthToken, token_key=token)
        user = auth_token.user

        document = get_object_or_404(Document, vector_id=vector_id, tenant=user.tenant)
        documents = retrieve_documents_by_vector_id(vector_id)
        if not documents:
            return Response({"error": "No documents found."}, status=status.HTTP_404_NOT_FOUND)
        return Response({"vector_id": vector_id,"documents": documents}, status=status.HTTP_200_OK)


class DeleteDocumentAPIView(APIView):
    def delete(self, request, token, vector_id):
        auth_token = get_object_or_404(AuthToken, token_key=token)
        user = auth_token.user
        document = get_object_or_404(Document, vector_id=vector_id, tenant=user.tenant)

        try:
            delete_documents_by_vector_id(vector_id)
            document.delete()
            return Response({"message": "Document deleted successfully."}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        

@api_view(['GET', 'POST', 'DELETE'])
def chat_history(request, token,vector_id):
    auth_token = get_object_or_404(AuthToken, token_key=token)
    user = auth_token.user
    if not vector_id:
        return Response({'error': 'Vector ID not provided.'}, status=status.HTTP_400_BAD_REQUEST)
    
    if not user:
        return Response({'error': 'User not authenticated.'}, status=status.HTTP_401_UNAUTHORIZED)

    try:
        chat = ChatHistory.objects.get(vector_id=vector_id)
    except ChatHistory.DoesNotExist:
        chat = None

    if request.method == 'GET':
        if chat:
            return Response({'history': chat.history})
        else:
            return Response({'history': []})

    elif request.method == 'POST':
        history = request.data.get('history', [])
        if chat:
            chat.history = history
            chat.save()
        else:
            ChatHistory.objects.create(vector_id=vector_id, history=history)
        return Response({'message': 'Chat history saved.'})

    elif request.method == 'DELETE':
        if chat:
            chat.delete()
        return Response({'message': 'Chat history cleared.'})
    

class MultiFileAskAPIView(generics.CreateAPIView):
    serializer_class = AskQuestionSerializer

    def post(self, request, token):
        auth_token = get_object_or_404(AuthToken, token_key=token)
        user = auth_token.user
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            question = serializer.validated_data['question']
            vector_ids = request.data.get('vector_ids', [])
            chat_history = serializer.validated_data.get('chat_history')
            user_identifier = serializer.validated_data['user_identifier']

            if not question:
                return Response({"error": "A question must be provided."}, status=status.HTTP_400_BAD_REQUEST)
            if not vector_ids:
                return Response({"error": "vector_ids are required."}, status=status.HTTP_400_BAD_REQUEST)

            # Validate document access
            for vector_id in vector_ids:
                document = get_object_or_404(Document, vector_id=vector_id, tenant=user.tenant)
                if user_identifier != user.email:  # Admin uses email as identifier
                    if not DocumentAccess.objects.filter(document=document, user_identifier=user_identifier).exists():
                        return Response({'error': f'No access to document {vector_id}.'}, status=status.HTTP_403_FORBIDDEN)

            try:
                documents = retrieve_documents_by_vector_ids(vector_ids)
                if not documents:
                    return Response({"error": "No documents found for the given vector_ids."}, status=status.HTTP_404_NOT_FOUND)

                answer = ask_question(
                    question,
                    source_type="file",
                    documents=documents,
                    chat_history=chat_history
                )

                # Update or create session
                sorted_vector_ids = sorted(vector_ids)
                vector_hash = hashlib.sha256(json.dumps(sorted_vector_ids).encode('utf-8')).hexdigest()
                session, created = MultiFileChatSession.objects.get_or_create(
                    vector_hash=vector_hash,
                    user_identifier=user_identifier,
                    tenant=user.tenant,
                    defaults={
                        'session_id': str(uuid.uuid4()),
                        'user': user,
                        'vector_ids': vector_ids,
                        'history': [{'question': question, 'answer': answer}]
                    }
                )

                if not created:
                    session.history.append({'question': question, 'answer': answer})
                    session.save()

                return Response({
                    'session_id': session.session_id,
                    'answer': answer
                }, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class GlobalAskAPIView(generics.CreateAPIView):
    serializer_class = AskQuestionSerializer

    def post(self, request, token):
        auth_token = get_object_or_404(AuthToken, token_key=token)
        user = auth_token.user
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            question = serializer.validated_data['question']
            chat_history = serializer.validated_data.get('chat_history')
            user_identifier = serializer.validated_data['user_identifier']

            if not question:
                return Response({"error": "A question must be provided."}, status=status.HTTP_400_BAD_REQUEST)

            try:
                documents = Document.objects.filter(tenant=user.tenant)
                vector_ids = [doc.vector_id for doc in documents]
                if not vector_ids:
                    return Response({"error": "No documents found for this tenant."}, status=status.HTTP_404_NOT_FOUND)

                retrieved_docs = retrieve_documents_by_vector_ids(vector_ids)
                answer = ask_question(
                    question,
                    source_type="file",
                    documents=retrieved_docs,
                    chat_history=chat_history
                )

                # After vector retrieval, also run table and SQL queries across all tenant docs
                vector_answer = answer
                table_answer = []
                sql_answer = []
                from pathlib import Path
                from .utils import handle_table_query
                for doc in Document.objects.filter(tenant=user.tenant):
                    ext = Path(doc.title).suffix.lower()
                    # Table query for CSV/Excel
                    if ext in ['.csv', '.xls', '.xlsx']:
                        try:
                            res = handle_table_query(doc.content, question)
                            if res is not None:
                                # normalize to list
                                items = res if isinstance(res, list) else [s.strip() for s in str(res).split(',')]
                                table_answer.extend(items)
                        except Exception:
                            pass
                    # SQL substring fallback
                    try:
                        ext_doc = ExternalDocument.objects.using('external').get(vector_id=doc.vector_id)
                        txt = ext_doc.content.lower()
                        ql = question.lower().strip()
                        if ql in txt:
                            idx = txt.find(ql)
                            snippet = ext_doc.content[max(idx-50,0):min(idx+150,len(txt))]
                            sql_answer.append(snippet)
                    except Exception:
                        pass
                # Deduplicate
                table_answer = list(dict.fromkeys(table_answer))
                sql_answer = list(dict.fromkeys(sql_answer))
                return Response({
                    'vector_answer': vector_answer,
                    'table_answer': table_answer or None,
                    'sql_answer': sql_answer
                }, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET'])
def get_document_alerts(request, vector_id):
    document = get_object_or_404(Document, vector_id=vector_id)
    alerts = DocumentAlert.objects.filter(document=document)
    alert_data = [{"keyword": alert.keyword, "snippet": alert.snippet, "created_at": alert.created_at} for alert in alerts]
    return Response({"alerts": alert_data}, status=status.HTTP_200_OK)

def compute_vector_hash(vector_ids: list[str]) -> str:
    sorted_ids = sorted(vector_ids)
    return hashlib.sha256(",".join(sorted_ids).encode()).hexdigest()


# This endpoint is for saving and retrieving chat history for multiple files
@api_view(['GET', 'POST', 'DELETE'])
def chat_history(request, token, vector_id):
    auth_token = get_object_or_404(AuthToken, token_key=token)
    user = auth_token.user
    user_identifier = request.query_params.get('user_identifier') or request.data.get('user_identifier')

    if not user_identifier:
        return Response({'error': 'user_identifier is required.'}, status=status.HTTP_400_BAD_REQUEST)

    # Validate document access
    document = get_object_or_404(Document, vector_id=vector_id, tenant=user.tenant)
    if user_identifier != user.email:  # Admin uses email as identifier
        if not DocumentAccess.objects.filter(document=document, user_identifier=user_identifier).exists():
            return Response({'error': 'No access to this document.'}, status=status.HTTP_403_FORBIDDEN)

    try:
        chat = ChatHistory.objects.get(vector_id=vector_id, user_identifier=user_identifier, tenant=user.tenant)
    except ChatHistory.DoesNotExist:
        chat = None

    if request.method == 'GET':
        if chat:
            return Response({'history': chat.history}, status=status.HTTP_200_OK)
        return Response({'history': []}, status=status.HTTP_200_OK)

    elif request.method == 'POST':
        history = request.data.get('history', [])
        if chat:
            chat.history = history
            chat.save()
        else:
            chat = ChatHistory.objects.create(
                vector_id=vector_id,
                user_identifier=user_identifier,
                history=history,
                tenant=user.tenant
            )
        return Response({'message': 'Chat history saved.'}, status=status.HTTP_200_OK)

    elif request.method == 'DELETE':
        if chat:
            chat.delete()
            return Response({'message': 'Chat history cleared.'}, status=status.HTTP_200_OK)
        return Response({'message': 'No chat history found.'}, status=status.HTTP_200_OK)


@api_view(['GET', 'POST', 'DELETE'])
def chat_history_multifile(request, token):
    auth_token = get_object_or_404(AuthToken, token_key=token)
    user = auth_token.user
    user_identifier = request.query_params.get('user_identifier') or request.data.get('user_identifier')

    if not user_identifier:
        return Response({'error': 'user_identifier is required.'}, status=status.HTTP_400_BAD_REQUEST)

    if request.method == 'POST':
        vector_ids = request.data.get('vector_ids', [])
        history = request.data.get('history', [])

        if not vector_ids:
            return Response({'error': 'vector_ids are required.'}, status=status.HTTP_400_BAD_REQUEST)

        # Validate document access
        for vector_id in vector_ids:
            document = get_object_or_404(Document, vector_id=vector_id, tenant=user.tenant)
            if user_identifier != user.email:  # Admin uses email as identifier
                if not DocumentAccess.objects.filter(document=document, user_identifier=user_identifier).exists():
                    return Response({'error': f'No access to document {vector_id}.'}, status=status.HTTP_403_FORBIDDEN)

        # Generate vector_hash
        sorted_vector_ids = sorted(vector_ids)
        vector_hash = hashlib.sha256(json.dumps(sorted_vector_ids).encode('utf-8')).hexdigest()

        # Create or update session
        session, created = MultiFileChatSession.objects.get_or_create(
            vector_hash=vector_hash,
            user_identifier=user_identifier,
            tenant=user.tenant,
            defaults={
                'session_id': str(uuid.uuid4()),
                'user': user,
                'vector_ids': vector_ids,
                'history': history
            }
        )

        if not created:
            session.vector_ids = vector_ids
            session.history = history
            session.save()

        return Response({'session_id': session.session_id, 'message': 'Session saved.'}, status=status.HTTP_200_OK)

    elif request.method == 'GET':
        session_id = request.query_params.get('session_id')
        if not session_id:
            return Response({'error': 'session_id is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            session = MultiFileChatSession.objects.get(
                session_id=session_id,
                user_identifier=user_identifier,
                tenant=user.tenant
            )
            return Response(MultiFileChatSessionSerializer(session).data, status=status.HTTP_200_OK)
        except MultiFileChatSession.DoesNotExist:
            return Response({'error': 'Session not found.'}, status=status.HTTP_404_NOT_FOUND)

    elif request.method == 'DELETE':
        session_id = request.data.get('session_id')
        if not session_id:
            return Response({'error': 'session_id is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            session = MultiFileChatSession.objects.get(
                session_id=session_id,
                user_identifier=user_identifier,
                tenant=user.tenant
            )
            session.delete()
            return Response({'message': 'Session deleted.'}, status=status.HTTP_200_OK)
        except MultiFileChatSession.DoesNotExist:
            return Response({'error': 'Session not found.'}, status=status.HTTP_404_NOT_FOUND)

class ShareDocumentAPIView(APIView):
    # permission_classes = [IsAdmin]  # Uncomment when RBAC is implemented
    def post(self, request, token):
        auth_token = get_object_or_404(AuthToken, token_key=token)
        user = auth_token.user
        print(user)

        vector_id = request.data.get('vector_id')
        user_identifier = request.data.get('user_identifier')  # e.g., TM email

        if not vector_id or not user_identifier:
            return Response({"error": "vector_id and user_identifier are required."}, status=status.HTTP_400_BAD_REQUEST)

        document = get_object_or_404(Document, vector_id=vector_id, tenant=user.tenant)

        access, created = DocumentAccess.objects.get_or_create(
            document=document,
            user_identifier=user_identifier,
            defaults={'granted_by': user}
        )

        if not created:
            return Response({"message": "Access already granted."}, status=status.HTTP_200_OK)

        return Response({"message": f"Access granted to {user_identifier} for document {vector_id}."}, status=status.HTTP_201_CREATED)
    
    def get(self, request, token):
        auth_token = get_object_or_404(AuthToken, token_key=token)
        user = auth_token.user

        user_identifier = request.data.get('user_identifier')  # e.g., TM email

        if not user_identifier:
            return Response({"error": "user_identifier is required."}, status=status.HTTP_400_BAD_REQUEST)

        # Query shared documents
        access_records = DocumentAccess.objects.filter(
            user_identifier=user_identifier,
            document__tenant=user.tenant
        )
        documents = [access.document for access in access_records]
        
        if not documents:
            return Response({"message": "No shared documents found.", "shared_documents": []}, status=status.HTTP_200_OK)

        serializer = DocumentSerializer(documents, many=True)
        return Response({
            "user_identifier": user_identifier,
            "shared_documents": serializer.data
            }, status=status.HTTP_200_OK)


class RemoveDocumentAccessAPIView(generics.DestroyAPIView):

    def put(self, request, token):
        auth_token = get_object_or_404(AuthToken, token_key=token)
        user = auth_token.user
        vector_id = request.data.get('vector_id')
        user_identifier = request.data.get('user_identifier')

        if not vector_id or not user_identifier:
            return Response({"error": "vector_id and user_identifier are required."}, status=status.HTTP_400_BAD_REQUEST)

        # Validate document access
        document = get_object_or_404(Document, vector_id=vector_id, tenant=user.tenant)

        try:
            access = DocumentAccess.objects.get(document=document, user_identifier=user_identifier)
            access.delete()
            # Update MultiFileChatSession to remove vector_id
            self.update_multi_file_session(user_identifier, vector_id, user.tenant)
            return Response({"message": "Access removed successfully for document {vector_id}."}, status=status.HTTP_200_OK)
        except DocumentAccess.DoesNotExist:
            return Response({"error": "Access does not exist."}, status=status.HTTP_400_BAD_REQUEST)
        
    def update_multi_file_session(self,user_identifier, vector_id, tenant):
        """Remove vector_id from MultiFileChatSession and update vector_hash."""
        sessions = MultiFileChatSession.objects.filter(user_identifier=user_identifier, tenant=tenant)
        for session in sessions:
            if vector_id in session.vector_ids:
                session.vector_ids.remove(vector_id)
                if session.vector_ids:  # Update hash if vector_ids is not empty
                    session.vector_hash = hashlib.sha256(
                        ''.join(sorted(session.vector_ids)).encode('utf-8')
                    ).hexdigest()
                else:  # Clear hash if no vector_ids remain
                    session.vector_hash = None
                session.save()