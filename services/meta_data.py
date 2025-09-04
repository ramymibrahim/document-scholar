class MetaDataService:
    def __init__(self, db, categories):
        self.db = db
        self.categories = categories

    def get_categories(self):
        return self.categories

    def get_search_paths(self):
        return self.db.get_rows("SELECT DISTINCT folder FROM files ORDER BY folder ASC")
