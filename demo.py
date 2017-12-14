from Behavior_Labels import Users
import pandas as pd
import time
if __name__ == '__main__':
    database = 'Yichuang'
    log_num = 1
    file = pd.read_excel('ID.xls')
    print(99)
    '''ids = [156074129,
            156072805,
            156070640,
            156069132,
            156068272,
            156066228,
            156062079,
            151002416,
            109023075,
            108052368]'''


    users = Users(database=database, endtime='20171026', fromcsv=True)
    ad_khhs = set(file['FUNDSACCOUNT'])
    custids = users.custids
    df = ad = pd.DataFrame(columns=users.labels)
    print(len(custids))
    st = time.time()
    i = 0
    for custid in ids:
        users.get_logdata(custid)
        if len(users.logasset) < log_num:
            continue
        '''users.abnormals_l(custid)
        users.high_shares_l(custid)
        users.holdings(custid, '20171026')
        users.holding_float(custid)
        users.hold_var(custid)
        users.hold_concept_var(custid)i
        
        users.limit_perference(custid)'''

        dic = users.get_labels(custid)
        ad = ad.append(dic, ignore_index=True)
        i += 1
        print(i)
    print((time.time() - st) / i)
    ad = ad.to_csv('result/advanced2.csv')
