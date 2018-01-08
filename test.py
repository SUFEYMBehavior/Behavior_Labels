"""
@version: ??
@author: Antastsy
@time: 18-1-2 
"""
import pandas
from Behavior_Labels import Users
if __name__ == '__main__':
    users = Users(database='test', endtime='20171026', fromcsv=False)
    custids = pandas.read_csv('datas/custid.csv')

    for custid in set(custids['custid']):
        users.get_logdata(custid)