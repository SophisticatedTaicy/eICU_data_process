import psycopg2
from param import *

class PostgreSQL:

    def __init__(self):
        self.database = DATABASE
        self.user = USER
        self.password = PASSWORD
        self.host = HOST
        self.port = PORT
        self.search_path = SEARCH_PATH
        self.connection = psycopg2.connect(
            database=self.database,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port)

    def get_list_by_id(self, id):
        cursor = self.connection.cursor()
        sql = "set search_path to " + self.search_path + ";"
        cursor.execute(sql)
        find_sql = "SELECT * FROM lab_pf WHERE ID=" + str(id) + ";"
        cursor.execute(find_sql)
        result = cursor.fetchall()
        cursor.close()
        self.connection.close()
        return result

    def get_apache4_by_id(self, id):
        cursor = self.connection.cursor()
        sql = "set search_path to " + self.search_path + ";"
        cursor.execute(sql)
        # find_sql = "SELECT * FROM lab_pf WHERE ID=" + str(id) + ";"
        find_sql = "SELECT apachescore from apachepatientresult WHERE patientunitstayid=" + str(id) + ";"
        cursor.execute(find_sql)
        result = cursor.fetchall()
        cursor.close()
        self.connection.close()
        return result

if __name__ == '__main__':
    print(PostgreSQL().get_list_by_id(141168))