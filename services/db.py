import sqlite3, pathlib


class Db:
    def __init__(self, sqlDbPath):
        self.sqlDbPath = sqlDbPath

    def get_rows(self, sql, parameters=[]):
        rows = []
        try:
            conn = sqlite3.connect(pathlib.Path(self.sqlDbPath))
            conn.row_factory = sqlite3.Row

            with conn:
                cur = conn.execute(sql, parameters)
                for row in cur:
                    rows.append(row)
        except ValueError as err:
            print("Value problem:", err)
        return [dict(r) for r in rows]

    def get_row_or_default(self, sql,parameters=[]):
        rows = self.get_rows(sql,parameters)
        if len(rows) == 0:
            return None
        return dict(rows[0])

    def execute(self, sql, parameters=[]):
        conn = sqlite3.connect(pathlib.Path(self.sqlDbPath))
        conn.row_factory = sqlite3.Row
        with conn:
            conn.execute(sql, parameters)
        return True

    def execute_many(self, sql, parameters=[]):
        conn = sqlite3.connect(pathlib.Path(self.sqlDbPath))
        conn.row_factory = sqlite3.Row
        with conn:
            conn.executemany(sql, parameters)
        return True
