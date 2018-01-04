"""
@version: ??
@author: Antastsy
@time: 18-1-2 
"""
from utility import MSSQL
if __name__ == '__main__':
    ms = MSSQL(host="localhost", user="SA", pwd="!@Cxy7300", db='a')
    sl = ms.ExecQuery("select * from a")
    print(sl)
    pass