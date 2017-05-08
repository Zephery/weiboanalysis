import pymysql

connection = pymysql.connect(host="123.206.28.24", user="root", passwd="root", db="weibo", port=3306,
                             charset="utf8")
cur = connection.cursor()
cur.execute("select * from weibo.data order by id desc limit 200,100")
result=cur.fetchall()
for i in result:
    print(i[5])
cur.close()
connection.close()

