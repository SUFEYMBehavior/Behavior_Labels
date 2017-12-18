from Behavior_Labels import Users
import pandas as pd
import time
if __name__ == '__main__':
    database = 'Yichuang'
    log_num = 60
    file = pd.read_excel('ID.xls')

    ids = [156074129,
            156072805,
            156070640,
            156069132,
            156068272,
            156066228,
            156062079,
            151002416,
            109023075,
            108052368]


    users = Users(database=database, endtime='20171026', fromcsv=False)
    ad_khhs = set(file['FUNDSACCOUNT'])

    df = ad = pd.DataFrame(columns=users.labels)
    #print(len(custids))
    st = time.time()
    i = 0
    for custid in ad_khhs:
        log = users.get_logdata(custid)
        if len(log)<1:
            continue
        dic = users.get_labels(custid)
        #if log>log_num:
        #    dic = users.get_GDZX_l(custid)
        i += 1
        ad = ad.append(dic, ignore_index=True)
        print(i)
    print((time.time() - st) / i)
    ad = ad.to_csv('advanced_users_noindex.csv',index=False)
