class ExternalDbRouter:
    """
    Routes ExternalDocument model operations to the 'external' database.
    """
    def db_for_read(self, model, **hints):
        if model.__name__ == "ExternalDocument":
            return 'external'
        return None

    def db_for_write(self, model, **hints):
        if model.__name__ == "ExternalDocument":
            return 'external'
        return None

    def allow_relation(self, obj1, obj2, **hints):
        if obj1.__class__.__name__ == "ExternalDocument" or obj2.__class__.__name__ == "ExternalDocument":
            return True
        return None

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        # Only allow ExternalDocument model in 'external' DB
        if model_name == 'externaldocument':
            return db == 'external'
        if db == 'external':
            return False
        return True