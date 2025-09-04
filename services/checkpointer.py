from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import sqlite3
from contextlib import closing

class CheckPointer:
    def __init__(self,path):
        self.path=path
        self.checkpointer_cm = AsyncSqliteSaver.from_conn_string(path)   
        self.checkpointer=None
                
    def delete_thread(self,thread_id):             
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA foreign_keys=ON")
        cur = conn.cursor()
        cur.execute("SELECT checkpoint_id FROM checkpoints WHERE thread_id = ?", (thread_id,))
        cids = [row[0] for row in cur.fetchall()]
        cid_tuples = tuple(cids) if cids else tuple()

        try:
            conn.execute("BEGIN")
            cur.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))

            if cids:
                q_marks = ",".join(["?"] * len(cids))
                cur.execute(f"DELETE FROM writes WHERE checkpoint_id IN ({q_marks})", cid_tuples)
            cur.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        conn.execute("VACUUM")

        conn.close()