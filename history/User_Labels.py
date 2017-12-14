import tushare as ts
import pandas as pd
import datetime
import pickle
from utility import *
import numpy as np
import time
import os



class user_label():
    def get_labels(self, custid):
        #self.get_dicts()

        zys = self.get_ZYZS_l(custid)
        kli = self.get_Kline_l(custid)[custid]
        abn = self.abnormals_l(custid)
        shr = self.high_shares_l(custid)
        self.get_dicts()
        try:
            phw = self.phwt_dict[custid]
        except:
            phw = [-1, -1]
        '''bgp = self.get_BGPHcustid(custid)
        tor = self.tor_pref(custid)
        fdp = self.fund_pref(custid)
        otd = self.otstd_pref(custid)
        opp = self.op_pref(custid)
        pep = self.pe_pref(custid)
        pbp = self.pb_pref(custid)
        npr = self.npr_pref(custid)
        top = self.top_pref(custid)
        mcd = self.macd_pref(custid)
        kdj = self.kdj_pref(custid)

        idp = self.industry_pref(custid)
        ccp = self.concept_pref(custid)
        id_tlt = pd.read_csv('concept_title.csv')
        cp_tlt = pd.read_csv('industry_title.csv')

        tlt = np.nonzero(idp[custid])[0]
        cp = np.nonzero(ccp[custid])[0]
        if len(tlt) == 0:
            tl = 'None'
        else:
            tl = [id_tlt['name'].iloc[i] for i in tlt]
        if len(cp) == 0:
            cpp = 'None'
        else:
            cpp = [id_tlt['name'].iloc[i] for i in cp]'''
        dic = {#'行业偏好': tl,
            #'概念偏好': cpp,
            '异常波动股偏好': abn['Q']*self.Q + abn['R']*self.R,
            '高转送股偏好': shr['Q']*self.Q + shr['R']*self.R,
            # '活跃股偏好': self.hyg_dict[str(custid)],
            '客户号': custid,
            '持仓概念偏好': self.hold_concept_var(custid),
            '持仓浮盈浮亏': self.holding_float(custid),
            '持股周期': self.holdings(custid, end_time=self.ed),
            '持仓偏好': self.hold_var(custid),
            '涨跌停操作偏好':  self.limit_perference(custid),
            #'委托方式偏好': self.get_WTFS_l(custid),
            '止盈偏好': zys[0],            '止损偏好': zys[1],
            'K线图': '',
            '日K买入': kli[0],  '周K买入': kli[1], '月K买入': kli[2],
            '日K卖出': kli[3],  '周K卖出': kli[4], '月K卖出': kli[5],
            '波段操作偏好': self.get_BDCZ_l(str(custid)),
            'bs操作偏好': self.bs_operate_l(custid),
            '过度自信': self.get_GDZX_l(custid),
            '偏好市价委托': phw[0],
            '偏好限价委托': phw[1],
        }

        #'委托渠道偏好': [wtq[1], wtq[2], wtq[4], wtq[8], wtq[16], wtq[32], wtq[64], wtq[128]],
        #'交易时间偏好': self.get_JYSJ(custid).iloc[0],
        #'打新股情况': self.get_XGQK(custid)}
        #'动量策略偏好': self.get_DLCL(custid).iloc[0],
        #'分时图': '',
        #'高点买入':fst['highB'].iloc[0],  '低点买入':fst['lowB'].iloc[0],  '上行买入':fst['upB'].iloc[0],  '下跌买入':fst['downB'].iloc[0],
        #'高点卖出': fst['highS'].iloc[0], '低点卖出':fst['lowS'].iloc[0],  '上行卖出':fst['upS'].iloc[0],  '下跌卖出':fst['downS'].iloc[0]
        #'新股偏好': self.get_XGPH(custid),
        #'可转债偏好': self.get_KZPH(custid),
        #'次新股偏好': self.get_CXPH(custid),
        #'ST股偏好': self.get_STPH(custid),
        #'B股偏好': (bgp['FB'], bgp['B']),
        #
        '''
        '偏好低换手率': tor[custid][0],
        '偏好中换手率': tor[custid][1],
        '偏好高换手率': tor[custid][2],
        '基金重仓股偏好': fdp[custid][0]*self.Q+fdp[custid][1]*self.R,
        '偏好高市值': otd[custid][0],
        '偏好中市值': otd[custid][1],
        '偏好低市值': otd[custid][2],
        '偏好低价股': opp[custid][0],
        '偏好中价股': opp[custid][1],
        '偏好高价股': opp[custid][2],
        '偏好低市净率': pbp[custid][2],
        '偏好中市净率': pbp[custid][1],
        '偏好高市净率': pbp[custid][0],
        '偏好高市盈率': pep[custid][2],
        '偏好中市盈率': pep[custid][1],
        '偏好低市盈率': pep[custid][0],
        '偏好正利润': npr[custid][0],
        '偏好负利润': npr[custid][1],
        '热点偏好': top[custid][0],
        '非热点偏好': top[custid][1],
        'MACD': mcd[custid],
        'KDJ': kdj[custid],
        # '行业偏好': [id_tlt.iloc[i, 0] for i in list(np.nonzero(idp[custid]))],
        # '概念偏好': [cp_tlt.iloc[i, 0] for i in list(np.nonzero(ccp[custid]))]]
        '''
        return dic
    def __init__(self, q=0.5, r=0.5, file='twtls2.xls'):
        self.path = './Customer/'
        self.file = file
        self.khhs = self.get_KHH(self.file)
        self.list_catalog = os.listdir(self.path)
        self.Q = q
        self.R = r
        self.X_flag = 0
        self.X_sta = 0.00
        self.S_flag = 0
        self.S_sta = 0.00
        self.C_flag = 0
        self.C_sta = 0.00
        self.K_flag = 0
        self.K_sta = 0.00

        try:
            self.phwt_dict = pickle.load(open('dict/phwt_dict', 'rb'))
        except:
            self.phwt_dict = self.get_PHWT_l()
            pickle.dump(self.phwt_dict, open('dict/phwt_dict', 'wb'))

        try:
            self.hyg_dict = pickle.load(open('dict/hyg_dict', 'rb'))
        except:
            self.hyg_dict = self.get_HYG_l()
            pickle.dump(self.hyg_dict, open('dict/hyg_dict', 'wb'))

        try:
            self.abn_dict = pickle.load(open('dict/abn_dict', 'rb'))
        except:
            self.abn_dict = self.abnormals_l()
            pickle.dump(self.abn_dict, open('dict/abn_dict', 'wb'))

        try:
            self.hsr_dict = pickle.load(open('dict/hsr_dict', 'rb'))
        except:
            self.hsr_dict = self.high_shares_l()
            pickle.dump(self.hsr_dict, open('dict/hsr_dict', 'wb'))




    def get_labels(self, khh):
        abn = self.abn_dict[self.abn_dict['khh'] == khh].iloc[0]# DATAFRAME Q&R
        shr = self.hsr_dict[self.hsr_dict['khh'] == khh].iloc[0]
        hyg = self.hyg_dict[str(khh)]
        phw = self.phwt_dict[str(khh)]
        cpt = self.hold_concept_l(khh)
        hft = self.holding_float_l(khh)
        hld = self.holdings(khh)
        htr = self.hold_trend_l(khh)
        lpr = self.limit_perference(khh)

        wtp = self.get_WTFS_l(khh)
        zys = self.get_ZYZS_l(khh)
        
        kli = self.get_Kline_l(str(khh))
        bdc = self.get_BDCZ_l(str(khh))
        wtq = self.get_WTQD(khh)
        jys = self.get_JYSJ(khh)
        xye = self.get_XGQK(khh)
        dls = self.get_DLCL(khh)
        
        fst = self.get_FSTPH(khh)
        xgp = self.get_XGPH(khh)
        kzz = self.get_KZPH(khh)
        cxg = self.get_CXPH(khh)
        stg = self.get_STPH(khh)
        bgp = self.get_BGPHKHH(khh)
        tor = self.tor_pref(khh)
        fdp = self.fund_pref(khh)
        otd = self.otstd_pref(khh)
        opp = self.op_pref(khh)
        pep = self.pe_pref(khh)
        pbp = self.pb_pref(khh)
        npr = self.npr_pref(khh)
        top = self.top_pref(khh)
        mcd = self.macd_pref(khh)
        kdj = self.kdj_pref(khh)
        idp = self.industry_pref(khh)
        ccp = self.concept_pref(khh)


        dic = {
            '异常波动股偏好': abn['Q']*self.Q + abn['R']*self.R,
            '高转送股偏好': shr['Q']*self.Q + shr['R']*self.R,
            '活跃股偏好': hyg,
            '持仓概念偏好': cpt,
            '持仓浮盈浮亏': hft,
            '持股周期': hld,
            '持仓偏好': htr,
            '涨跌停操作偏好': lpr,
            '委托方式偏好': wtp,
            '止盈偏好': zys[0],            '止损偏好': zys[1],
            'K线图偏好': kli[str(khh)],
            '波段操作偏好': bdc[str(khh)],
            '委托渠道偏好': [wtq[1], wtq[2], wtq[4], wtq[8], wtq[16], wtq[32], wtq[64], wtq[128]],
            '交易时间偏好': jys.iloc[0],
            '打新股情况': xye['lucky'],
            '动量策略偏好': dls.iloc[0],
            '分时图偏好': fst.iloc[0],
            '新股偏好': xgp['XGPH'],
            '可转债偏好': kzz['KZZ'],
            '次新股偏好': cxg['CX'],
            'ST股偏好': stg['ST'],
            'B股偏好': (bgp['FB'], bgp['B']),
            '偏好市价委托': phw[0],
            '偏好限价委托': phw[1],
            '换手率偏好': tor[khh],
            '基金重仓股偏好': fdp[khh][0]*self.Q+fdp[khh][1]*self.R,
            '主题热点偏好': otd[khh],
            '股价偏好': opp[khh],
            '市盈率偏好': pbp[khh],
            '市净率偏好': pep[khh],
            '净利润偏好': npr[khh],
            '热点偏好': top[khh][0]*self.Q+top[khh][1]*self.R,
            'MACD': mcd[khh],
            'KDJ': kdj[khh],
            '行业偏好': idp[khh],
            '概念偏好': ccp[khh]
        }
        return dic

    def get_concept(self):

        concept = pd.DataFrame(ts.get_concept_classified())
        print(concept.shape)
        concept.to_csv('concept.csv')

    def get_industry(self):
        industry = pd.DataFrame(ts.get_industry_classified())
        print(industry.shape)
        industry.to_csv('industry.csv')

    def abnormals_l(self, spl=1, path='datas/data_abnormal'):
        m = []
        try:
            f = open(path, 'rb')
            data = pickle.load(f)
            f.close()
        except:
            data = {}
        user = pd.read_excel(self.file)
        user = user[user["CJSL"] > 0]
        user = user.sort_values("WTRQ")
        for i, row in user.iterrows():
            t2 = str(int(row["WTRQ"]))
            time = datetime.datetime.strptime(t2, '%Y%m%d')
            st_time = time - datetime.timedelta(days=30)
            t1 = datetime.datetime.strftime(st_time, '%Y-%m-%d')
            t2 = datetime.datetime.strftime(time, '%Y-%m-%d')
            zqdm = str(int(row['ZQDM']))
            # 取前一个月的历史行情 如果文件中有则取 否则查询
            if (zqdm, t1, t2) in data.keys():
                his_data = data[(zqdm, t1, t2)]
            else:
                his_data = ts.get_hist_data(code=zqdm, start=t1, end=t2)
                data[(zqdm, t1, t2)] = his_data
                f = open(path, 'wb')
                pickle.dump(data, f)
                f.close()
            if his_data is not None and len(his_data['p_change']) > 0:
                p = his_data['p_change'].mean()
                m.append(p)
            else:
                m.append(0)
        dis = distribution(m)
        flag = [True if x > spl else False for x in dis]
        user['flag'] = flag
        dic = calculate(user, label='flag')
        return dic

    def hold_concept_l(self, khh, path='datas/industry.csv'):
        user = pd.read_excel(self.file)
        user = user[user['KHH'] == khh]
        user = user[user["CJSL"] > 0]
        user = user.sort_values("WTRQ")
        file = path
        concept = pd.read_csv(file)
        conc = {}
        for _, row in user.iterrows():
            zqdm = int(row["ZQDM"])
            rec = concept.loc[concept['code'] == zqdm]
            conc[zqdm] = list(rec['c_name'])
        # print(conc)

        khhs = [0, 0]
        holdings = pd.DataFrame(columns=['concept', 'num'])
        st_time = datetime.datetime.strptime(str(int(user.iloc[0]["WTRQ"])), '%Y%m%d')
        for _, row in user.iterrows():
            time = datetime.datetime.strptime(str(int(row["WTRQ"])), '%Y%m%d')
            while st_time < time:
                codehold = np.array(holdings['num'])
                if len(codehold) == 0:
                    return 0
                var = np.var(codehold / np.sum(codehold))
                khhs[0] += var
                khhs[1] += 1
                st_time += datetime.timedelta(days=1)
                # print(st_time)
                # print(holdings)
            wtlb = row["WTLB"]
            if wtlb in buyset:
                flag = 1
            elif wtlb in sellset:
                flag = 0
            else:
                continue
            zqdm = int(row["ZQDM"])
            if not zqdm in conc.keys():
                continue  # 证券代码没有概念分类 不作数
            concpt = conc[zqdm]
            if len(concpt) is 0:  # 当前证券不属于任何概念 跳过
                continue
            cjsl = int(row["CJSL"])
            for con in concpt:  # 遍历当前证券所有属于的概念
                record = holdings[holdings['concept'] == con]
                # 得到当前所属概念的持仓情况
                if len(record) is 0:
                    if flag is 0:  # 如果还未持仓而且该笔交易是卖出 不计入
                        continue
                    holdings = holdings.append({'concept': con, 'var': 0, 'num': cjsl}, ignore_index=True)
                else:
                    # 已有交易了 则修改记录中的字段
                    if flag:
                        record.iloc[0]['num'] += cjsl
                    else:
                        record.iloc[0]['num'] -= cjsl
                        if record.iloc[0]['num'] < 0:
                            record.iloc[0]['num'] = 0
                            # print(holdings)

        if khhs[1] == 0:
            return 0
        else:
            return khhs[0]/khhs[1]

    def holding_float_l(self, khh):
        user = pd.read_excel(self.file)
        user = user[user['KHH'] == khh]
        user = user[user["CJSL"] > 0]
        user = user.sort_values("WTRQ")
        khhs = 0
        float_holding = pd.DataFrame(columns=['code', 'cost', 'num', 'profit', 'price', 'days'])
        st_time = datetime.datetime.strptime(str(int(user.iloc[0]["WTRQ"])), '%Y%m%d')
        for _, row in user.iterrows():
            time = datetime.datetime.strptime(str(int(row["WTRQ"])), '%Y%m%d')
            codelist = set(float_holding['code'])
            while st_time < time:
                for code in codelist:
                    ctime = datetime.datetime.strftime(st_time, '%Y-%m-%d')
                    his = ts.get_hist_data(code=str(int(code)), start=ctime, end=ctime)
                    if his is None or len(his) is 0:
                        continue
                    else:
                        price = his.iloc[0]['close']
                        for i, hold in float_holding.iterrows():
                            if hold['code'] == code:
                                hold['profit'] += (price - hold['cost']) * hold['num']
                                hold['price'] = price
                st_time += datetime.timedelta(days=1)
                float_holding['days'] += 1

            wtlb = row["WTLB"]
            if wtlb in buyset:
                flag = 1
            elif wtlb in sellset:
                flag = 0
            else:
                continue
            zqdm = int(row["ZQDM"])
            cjje = row['CJJG']
            cjsl = int(row["CJSL"])
            if cjsl == 0:
                continue
            record = float_holding[float_holding['code'] == zqdm]
            if len(record) is 0:
                if flag is 0:
                    continue
                float_holding = float_holding.append({'code': zqdm, 'cost': cjje,
                                                      'num': cjsl, 'profit': 0, 'price': cjje, 'days': 0},
                                                     ignore_index=True)
            else:
                if flag:
                    record['cost'] = (record['cost'] * record.iloc[0]['num'] + cjje) / (record.iloc[0]['num'] + cjsl)
                    record.iloc[0]['num'] += cjsl
                else:
                    record.iloc[0]['num'] -= cjsl
                    if record.iloc[0]['num'] < 0:
                        record.iloc[0]['num'] = 0

        for _, row in float_holding.iterrows():
            if not row['days'] == 0:
                khhs += row['profit'] / row['days']
        return khhs

    def holdings(self, khh):
        user = pd.read_excel(self.file)
        user = user[user['KHH'] == khh]
        user = user[user["CJSL"] > 0]
        user = user.sort_values("WTRQ")
        avg_time = []
        for i, row in user.iterrows():
            cjsl = int(row["CJSL"])
            if cjsl == 0:
                continue
            if row["WTLB"] in buyset:
                flag = 1
            elif row["WTLB"] in sellset:
                flag = 0
            else:
                continue
            zqdm = int(row["ZQDM"])
            time = datetime.datetime.strptime(str(int(row["WTRQ"])), '%Y%m%d')

            holdstock = {}

            if zqdm not in holdstock.keys():
                holdstock[zqdm] = []
            holdqueue = holdstock[zqdm]
            if flag:  # 如果是买入则在队尾添加
                holdqueue.append((cjsl, time))
            else:
                for index, (num, t) in enumerate(holdqueue):
                    delta = time - t
                    if num is 0:
                        continue
                    if num >= cjsl:  # 当前这笔交易被“填平”了
                        num -= cjsl
                        avg_time.append((cjsl, delta.days))
                        holdqueue[index] = (num, t)
                        break
                    else:
                        avg_time.append((num, delta.days))
                        num = 0
                        cjsl -= num
                        holdqueue[index] = (num, t)
                        # print(holdqueue)
        times = nums = 1
        for num, time in avg_time:
            nums += num
            times += num * time
        return times/nums

    def hold_trend_l(self, khh):
        user = pd.read_excel(self.file)
        user = user[user['KHH'] == khh]
        user = user[user["CJSL"] > 0]
        user = user.sort_values("WTRQ")
        khhs = [0, 0.01]
        holdings = pd.DataFrame(columns=['code', 'num'])
        st_time = datetime.datetime.strptime(str(int(user.iloc[0]["WTRQ"])), '%Y%m%d')
        for _, row in user.iterrows():
            time = datetime.datetime.strptime(str(int(row["WTRQ"])), '%Y%m%d')
            while st_time < time:
                codehold = np.array(holdings['num'])
                if len(codehold) == 0:
                    return 0
                var = np.var(codehold / np.sum(codehold))
                khhs[0] += var
                khhs[1] += 1
                st_time += datetime.timedelta(days=1)
                # print(st_time)
                # print(holdings)
            wtlb = row["WTLB"]
            if wtlb in buyset:
                flag = 1
            elif wtlb in sellset:
                flag = 0
            else:
                continue
            zqdm = int(row["ZQDM"])
            cjsl = int(row["CJSL"])
            if cjsl == 0:
                continue
            record = holdings[holdings['code'] == zqdm]
            if len(record) is 0:
                if flag is 0:
                    continue
                holdings = holdings.append({'code': zqdm, 'var': 0, 'num': cjsl}, ignore_index=True)
            else:
                if flag:
                    record.iloc[0]['num'] += cjsl
                else:
                    record.iloc[0]['num'] -= cjsl
                    if record.iloc[0]['num'] < 0:
                        record.iloc[0]['num'] = 0

        return khhs[0]/khhs[1]

    def limit_perference(self, khh, time_range=1, path='datas/data_limit'):
        try:
            f = open(path, 'rb')
            data = pickle.load(f)
        except:
            data = {}
        khhs = [0, 0]
        user = pd.read_excel(self.file)
        user = user[user['KHH'] == khh]
        user = user[user["CJSL"] > 0]
        user = user.sort_values("WTRQ")
        for i, row in user.iterrows():
            cjsl = row["CJSL"]
            if cjsl == 0:
                continue
            cal_num = khhs[0]
            cal_all = khhs[1]
            t2 = str(int(row["WTRQ"]))
            time = datetime.datetime.strptime(t2, '%Y%m%d')
            st_time = time - datetime.timedelta(days=time_range)
            t1 = datetime.datetime.strftime(st_time, '%Y-%m-%d')
            t2 = datetime.datetime.strftime(time, '%Y-%m-%d')
            zqdm = str(int(row['ZQDM']))
            # 取前一个月的历史行情 如果文件中有则取 否则查询
            if (zqdm, time) in data.keys():
                p_change = data[(zqdm, time)]
            else:
                his_data = ts.get_hist_data(code=zqdm, start=t1, end=t2)
                if his_data is not None and len(his_data) > 0:
                    p_change = his_data['p_change'].max()
                    data[(zqdm, time)] = p_change
                    f = open(path, 'wb')
                    pickle.dump(data, f)
                    f.close()
                else:
                    p_change = 0.0
            cal_all += 1
            if p_change >= 9.95:
                cal_num += 1
            khhs = [cal_num, cal_all]

        return khhs[0]/khhs[1]

    def high_shares_l(self, path='datas/shares.xlsx'):
        file = open(path, 'rb')
        shr = pd.read_excel(file)
        user = pd.read_excel(self.file)
        user = user.sort_values("WTRQ")
        flag = []
        for _, row in user.iterrows():
            dm = row["ZQDM"]
            sh = shr[shr['code'] == dm]
            delta = 1000
            tm = 0
            sbrq = datetime.datetime.strptime(str(int(row["WTRQ"])), '%Y%m%d')
            for __, rw in sh.iterrows():
                tm1 = datetime.datetime.strptime(rw['report_date'], '%Y-%m-%d')
                days = sbrq - tm1
                if tm1 < sbrq and days.days < delta:
                    delta = days.days
                    tm = rw['shares']
            if tm > 5:
                flag.append(True)
            else:
                flag.append(False)
        user['flag'] = flag
        dic = calculate(user, 'flag')
        return dic

    def calculate_the_means_and_var(self, dict_name, col_number):
        nlist = []
        for d in dict_name.keys():
            if col_number == 'none':
                nlist.append(dict_name[d])
            elif dict_name[d][0] == -1:
                continue
            else:
                nlist.append(dict_name[d][col_number])
        narray = np.array(nlist)
        mean = narray.mean()
        var = narray.var()
        return mean, var

    def get_n_days_date(self, today_string, n):
        today_date = time.strptime(today_string, "%Y%m%d")  # 字符串转换成time类型
        # print type(date)  # 查看date的类型<type 'time.struct_time'>
        date = datetime.datetime(today_date[0], today_date[1], today_date[2])  # time类型转换成datetime类型
        day = date.strftime('%w')
        ntime = date - datetime.timedelta(days=n)
        time_string = ntime.strftime("%Y-%m-%d")
        return time_string

    # 委托渠道偏好(输入一个客户号即可得到标签，无需用到所有人的数据）
    def get_WTFS_l(self, khh):
        customer_file = str(khh) + ".txt"
        deep_catalog = os.path.join(self.path, customer_file)
        file = open(deep_catalog).readlines()
        wtfs_fre_list = []
        wtfs_dict = {}
        entrust_time = 0
        entrust_way1 = 0
        entrust_way2 = 0
        entrust_way4 = 0
        entrust_way8 = 0
        entrust_way16 = 0
        entrust_way32 = 0
        entrust_way64 = 0
        entrust_way128 = 0

        for line in file:
            entrust_way = line.split('\t')[56]
            entrust_time = entrust_time + 1
            if entrust_way == "1":
                entrust_way1 = entrust_way1 + 1
            elif entrust_way == "2":
                entrust_way2 = entrust_way2 + 1
            elif entrust_way == "4":
                entrust_way4 = entrust_way4 + 1
            elif entrust_way == "8":
                entrust_way8 = entrust_way8 + 1
            elif entrust_way == "16":
                entrust_way16 = entrust_way16 + 1
            elif entrust_way == "32":
                entrust_way32 = entrust_way32 + 1
            elif entrust_way == "64":
                entrust_way64 = entrust_way64 + 1
            else:
                entrust_way128 = entrust_way128 + 1

        entrust_way1_frequncy = float(entrust_way1 / entrust_time)
        wtfs_fre_list.append(entrust_way1_frequncy)
        entrust_way2_frequncy = float(entrust_way2 / entrust_time)
        wtfs_fre_list.append(entrust_way2_frequncy)
        entrust_way4_frequncy = float(entrust_way4 / entrust_time)
        wtfs_fre_list.append(entrust_way4_frequncy)
        entrust_way8_frequncy = float(entrust_way8 / entrust_time)
        wtfs_fre_list.append(entrust_way8_frequncy)
        entrust_way16_frequncy = float(entrust_way16 / entrust_time)
        wtfs_fre_list.append(entrust_way16_frequncy)
        entrust_way32_frequncy = float(entrust_way32 / entrust_time)
        wtfs_fre_list.append(entrust_way32_frequncy)
        entrust_way64_frequncy = float(entrust_way64 / entrust_time)
        wtfs_fre_list.append(entrust_way64_frequncy)
        entrust_way128_frequncy = float(entrust_way128 / entrust_time)
        wtfs_fre_list.append(entrust_way128_frequncy)
        wtfs_dict[khh] = wtfs_fre_list

        return wtfs_dict

    def Checktime(self, starttime, endtime, given_time):
        Flag = 0
        starttime = time.strptime(starttime, '%H:%M:%S')
        endtime = time.strptime(endtime, '%H:%M:%S')
        if (starttime < given_time) and (endtime > given_time):
            Flag = 1
        else:
            Flag = 0
        return Flag

    # 偏好委托  (返回的是集体的偏好委托，需要用到均值和方差）
    def get_PHWT_l(self):
        min_time_interval = datetime.datetime.strptime("00:00:00", "%H:%M:%S") - datetime.datetime.strptime("00:00:00",
                                                                                                            "%H:%M:%S")
        max_time_interval = datetime.datetime.strptime("00:00:03", "%H:%M:%S") - datetime.datetime.strptime("00:00:00",
                                                                                                            "%H:%M:%S")
        phwt_dict = {}
        for j in self.list_catalog:
            deep_catalog = os.path.join(self.path, j)
            file = open(deep_catalog).readlines()

            phwt_fre_list = []
            wt_times = 0
            sjwt_times = 0
            xjwt_times = 0
            khh = j.strip('.txt')
            for line in file:
                wt_time = line.split("\t")[26]
                #print(wt_time)
                sb_time = line.split("\t")[30]
                #print(sb_time)
                cj_time = line.split("\t")[43]
                #print(cj_time)
                wt_time = time.strptime(wt_time, "%H:%M:%S")
                time_flag_1 = self.Checktime("09:30:00", "11:30:00", wt_time)
                time_flag_2 = self.Checktime("13:00:00", "15:00:00", wt_time)
                if time_flag_1 == 0 and time_flag_2 == 0:
                    wt_times = wt_times + 1
                    xjwt_times = xjwt_times + 1
                elif cj_time == "00:00:00":
                    continue
                else:
                    sb_time = datetime.datetime.strptime(sb_time, "%H:%M:%S")
                    cj_time = datetime.datetime.strptime(cj_time, "%H:%M:%S")
                    time_interval = cj_time - sb_time
                    if time_interval < min_time_interval:
                        continue
                    elif time_interval < max_time_interval:
                        wt_times = wt_times + 1
                        sjwt_times = sjwt_times + 1
                    else:
                        wt_times = wt_times + 1
                        xjwt_times = xjwt_times + 1
            if int(wt_times) == 0:
                phwt_dict[khh] = [-1, -1]
            else:
                sjwt_fre = float(sjwt_times / wt_times)
                phwt_fre_list.append(sjwt_fre)
                xjwt_fre = float(xjwt_times / wt_times)
                phwt_fre_list.append(xjwt_fre)
                phwt_dict[khh] = phwt_fre_list

        sjwt_fre_mean, sjwt_fre_var = self.calculate_the_means_and_var(phwt_dict, 0)
        xjwt_fre_mean, xjwt_fre_var = self.calculate_the_means_and_var(phwt_dict, 1)
        for d in phwt_dict.keys():
            if phwt_dict[d][0] == -1:
                continue
            else:
                phwt_dict[d][0] = (phwt_dict[d][0] - sjwt_fre_mean) / sjwt_fre_var
                phwt_dict[d][1] = (phwt_dict[d][1] - xjwt_fre_mean) / xjwt_fre_var
                if phwt_dict[d][0] > phwt_dict[d][1]:
                    phwt_dict[d][0] = 1
                    phwt_dict[d][1] = 0
                else:
                    phwt_dict[d][0] = 0
                    phwt_dict[d][1] = 1

        return phwt_dict

    # 止盈止损(输入一个客户号即可得到标签，无需用到所有人的数据）
    def get_ZYZS_l(self, khh):
        customer_file = str(khh) + ".txt"
        deep_catalog = os.path.join(self.path, customer_file)
        file = open(deep_catalog).readlines()
        file = file[::-1]  # 将文件倒置
        zy_dict = {}
        zs_dict = {}
        dict = {}
        total_ylcjsl = 0
        total_kscjsl = 0
        total_ylv = 0
        total_ksl = 0
        for line in file:
            cjsl = line.split("\t")[38]
            wtlb = line.split("\t")[11]
            cjjg = float(line.split("\t")[42])
            if cjsl == "0":
                continue
            elif wtlb == '1':  # 委托类别为买入
                zqdm = line.split("\t")[14]
                cjsl = float(cjsl)
                if zqdm in dict.keys():  # 如果已存在该证券，则证券均值改变，股数改变
                    dict[zqdm][0] = (dict[zqdm][0] * dict[zqdm][1] + cjsl * cjjg) / (dict[zqdm][1] + cjsl)
                    dict[zqdm][1] = dict[zqdm][1] + cjsl
                else:  # 如果没有该证券，则新建一个字典
                    jy_list = []
                    jy_list.append(cjjg)
                    jy_list.append(cjsl)
                    dict[zqdm] = jy_list
            elif wtlb == "2":  # 委托类别为卖出
                zqdm = line.split("\t")[14]
                cjsl = float(cjsl)
                if zqdm in dict.keys():  # 已经有买入交易
                    jc = cjjg - dict[zqdm][0]
                    dict[zqdm][1] = dict[zqdm][1] - cjsl
                    if jc > 0:
                        ylv = jc / dict[zqdm][0]  # 单笔交易的盈利率
                        total_ylcjsl = total_ylcjsl + cjsl
                        total_ylv = ylv * cjsl + total_ylv
                    else:
                        ksl = -jc / dict[zqdm][0]  # 单笔交易的亏损率
                        total_kscjsl = total_kscjsl + cjsl
                        total_ksl = ksl * cjsl + total_ksl
                else:
                    continue  # 如果之前没有该股票的买入记录（数据时间问题）则忽视该笔交易
        if total_ylcjsl == 0:
            zyl = 0
        else:
            zyl = total_ylv / total_ylcjsl
        if total_kscjsl == 0:
            zsl = 0
        else:
            zsl = total_ksl / total_kscjsl
        zy_dict[khh] = zyl
        zs_dict[khh] = zsl
        return zyl, zsl

    # 活跃股（返回的是集体的偏好委托，需要用到均值和方差,应该套用q&r公式)
    def get_HYG_l(self):
        hyg_dict = {}
        for j in self.list_catalog:
            deep_catalog = os.path.join(self.path, j)
            file = open(deep_catalog).readlines()
            total_cjje = 0.01
            hyg_jyzj = 0
            khh = j.strip(".txt")
            for line in file:
                wtlb = line.split('\t')[11]
                cjsl = line.split("\t")[38]
                cjje = float(line.split("\t")[39])
                if (wtlb == "1") and (cjsl != "0"):
                    zqdm = line.split('\t')[14]
                    total_cjje = total_cjje + cjje
                    wtrq = line.split('\t')[25]
                    wtrq = self.get_n_days_date(wtrq, 0)
                    try:
                        gphq = ts.get_hist_data(zqdm, start=wtrq, end=wtrq)
                        hsl = float(gphq['turnover'][0])
                    except:
                        continue
                    if hsl > 0.02: hyg_jyzj = hyg_jyzj + cjje
                else:
                    continue
            hyg_ph = hyg_jyzj / total_cjje
            hyg_dict[khh] = hyg_ph

        # 标准化：
        mean, var = self.calculate_the_means_and_var(hyg_dict, 'none')
        for khh in hyg_dict.keys():
            hyg_dict[khh] = (hyg_dict[khh] - mean) / var
            if hyg_dict[khh] > 0.85:
                hyg_dict[khh] = 1  # 80%分位数
            else:
                hyg_dict[khh] = 0
        return hyg_dict

    # k_line(输入客户号即可返回，不用考虑其他用户)
    def get_Kline_l(self, khh):
        customer_file = str(khh) + ".txt"
        deep_catalog = os.path.join(self.path, customer_file)
        file = open(deep_catalog).readlines()
        kline_dict = {}
        kline_fre_list = []
        zgmr_times = 0
        zgmc_times = 0
        kdmr_times = 0
        kdmc_times = 0
        kzmr_times = 0
        kzmc_times = 0
        total_times = 0
        for line in file:
            zqdm = line.split('\t')[14]
            wtlb = line.split('\t')[11]
            wtrq = line.split('\t')[25]
            yesterday = self.get_n_days_date(wtrq, 10)
            tomorrow = self.get_n_days_date(wtrq, -10)
            wtrq = self.get_n_days_date(wtrq, 0)
            df1 = ts.get_hist_data(str(int(zqdm)), start=yesterday, end=wtrq)
            df2 = ts.get_hist_data(str(int(zqdm)), start=wtrq, end=tomorrow)
            if df1 is None or df2 is None or len(df1) == 0 or len(df2) == 0:
                continue
            yesterday_5ma = float(df1.ix[1]['ma5'])
            today_5ma = float(ts.get_hist_data(zqdm, start=wtrq, end=wtrq)['ma5'][0])

            tomarrow_5ma = float(df2.ix[-2]['ma5'])
            if wtlb == '1' and yesterday_5ma < today_5ma and tomarrow_5ma > today_5ma:
                zgmr_times = zgmr_times + 1
            elif wtlb == '1' and yesterday_5ma > today_5ma:
                kdmr_times = kdmr_times + 1
            elif wtlb == '1' and yesterday_5ma <= today_5ma:
                kzmr_times = kzmr_times + 1
            elif wtlb == '2' and yesterday_5ma < today_5ma and tomarrow_5ma > today_5ma:
                zgmc_times = zgmc_times + 1
            elif wtlb == '2' and yesterday_5ma > today_5ma:
                kdmc_times = kdmc_times + 1
            elif wtlb == '2' and yesterday_5ma <= today_5ma:
                kzmc_times = kzmc_times + 1
            total_times = total_times + 1

        zgmr_fre = zgmr_times / total_times
        zgmc_fre = zgmc_times / total_times
        kzmr_fre = kzmr_times / total_times
        kzmc_fre = kzmc_times / total_times
        kdmr_fre = kdmr_times / total_times
        kdmc_fre = kdmc_times / total_times

        kline_fre_list.append(zgmr_fre)
        kline_fre_list.append(zgmc_fre)
        kline_fre_list.append(kzmr_fre)
        kline_fre_list.append(kzmc_fre)
        kline_fre_list.append(kdmr_fre)
        kline_fre_list.append(kdmc_fre)

        kline_dict[khh] = kline_fre_list
        return kline_dict

    # 波段操作(输入客户号即可返回波段操作偏好）
    def get_CCI(self, df, N):
        df['typ'] = (df['high'] + df['low'] + df['close']) / 3
        df['cci'] = ((df['typ'] - df['typ'].rolling(N).mean()) /
                     (0.015 * abs(df['typ'] - df['typ'].rolling(N).mean()).rolling(N).mean()))
        return df

    def get_BDCZ_l(self, khh):
        customer_file = str(khh) + ".txt"
        deep_catalog = os.path.join(self.path, customer_file)
        file = open(deep_catalog).readlines()
        bdcz_dict = {}
        cci_times = 0
        wt_times = 0.01
        for line in file:
            cjsl = line.split("\t")[38]
            wtlb = line.split("\t")[11]
            cjjg = float(line.split("\t")[42])
            if cjsl == "0":
                continue
            elif wtlb == '1':  # 委托类别为买入
                zqdm = line.split("\t")[14]
                wtrq = line.split('\t')[25]
                start_time = self.get_n_days_date(wtrq, 0)
                end_time = self.get_n_days_date(wtrq, 50)
                gphq = ts.get_hist_data(str(int(zqdm)), start=end_time, end=start_time)
                if gphq is None or len(gphq)  == 0:
                    continue
                df = gphq[['open', 'high', 'low', 'close', 'volume']].sort_index(ascending=True)
                cci_df = self.get_CCI(df, 14)
                cci = cci_df.iloc[-1]['cci']
                if cci < -100: cci_times = cci_times + 1
            elif wtlb == '2':  # 委托类别为卖出
                zqdm = line.split("\t")[14]
                wtrq = line.split('\t')[25]
                start_time = self.get_n_days_date(wtrq, 0)
                end_time = self.get_n_days_date(wtrq, 50)
                gphq = ts.get_hist_data(str(int(zqdm)), start=end_time, end=start_time)
                if gphq is None or len(gphq)  == 0:
                    continue
                df = gphq[['open', 'high', 'low', 'close', 'volume']].sort_index(ascending=True)
                cci_df = self.get_CCI(df, 14)
                cci = cci_df.iloc[-1]['cci']
                if cci > 100: cci_times = cci_times + 1
            wt_times = wt_times + 1

        cci_fre = float(cci_times / wt_times)
        bdcz_dict[khh] = cci_fre
        return bdcz_dict

    # 得到所有客户号
    def get_KHH(self, file):
        datas = pd.read_excel(file)
        kh = list(set(np.array(datas["custid"]).tolist()))
        return kh

    # 委托方式需要根据标签数据
    # 委托渠道偏好
    '''
    def get_WTQD(self, KHH):
    	datas = pd.read_excel(self.file)
    	dirctory = {1: "tel", 2: "card", 4: "key", 8: "account", 16: "long", 32: "internet", 64: "phone", 128: "bank"} #存在问题!!!!!!!!!!!!!!!
    	l = dirctory.keys()
    	datas = datas.loc[:, ["custid", "operway"]]
    	res_k = {}
    	res_k["KHH"] = KHH
    	df = datas[datas["custid"] == KHH]
    	for w in l:
    		try:
    			n = (float(df[df["operway"] == w].shape[0]) / df.shape[0])
    		except:
    			n = (0 / df.shape[0])
    		res_w = n
    		res_k[w] = res_w
    	return res_k
    '''
    '''
    # 交易时间偏好
    def get_JYSJ(self, KHH):
    	datas = pd.read_excel(self.file)
    	N = ["custid", "bsflag", "ordertime", "market", "stktype","fundeffect"]
    	lb = {1: "ZPJH", 2: "ZPKH", 3: "ZPPZ", 4: "ZPSQ", 5: "ZPSH", 6: "XPKH", 7: "XPPZ", 8: "XPSQ", 9: "SPJH"}
    	tim = {1: "09:30:00", 2: "10:00:00", 3: "11:00:00", 4: "11:30:00", 5: "13:00:00", 6: "13:30:00", 7: "14:30:00",
    		   8: "15:00:00",
    		   9: "14:57:00"}
    	for t in range(9):  # 提前设定好每个标签类别对应的时间并解析成时间格式
    		tim[t + 1] = time.strptime(tim[t + 1], '%H:%M:%S')

    	datas = datas[datas["matchamt"] != 0.00]
    	datas = datas.loc[:, N]
    	res_al = []
    	dataB=datas[datas["fundeffect"]<0.00]
    	dataS=datas[datas["fundeffect"]>0.00]
    	alW=[dataB,dataS]
    	for data in alW:
    		if data["fundeffect"].iloc[0]<0.00:
    			way = "B"
    		else:
    			way = "S"
    		res_k = {}
    		res_k["KHH"] = int(KHH)
    		df = data[data["custid"] == int(KHH)]
    		aln = df.shape[0]
    		if aln == 0:
    			for j in range(9):
    				res_k[str(j + 1) + way] = 0.0000
    		else:
    			for i in range(aln):  # 遍历该用户的每条交易记录
    				trade_time = time.strptime(str(df.iloc[i, 3])[:-2], '%H%M%S')  # 解析该记录的用户交易时间
    				if (trade_time < tim[1]) & (trade_time >= tim[6]):
    					if str(1) + way not in res_k.keys():
    						res_k[str(1) + way] = 1
    					else:
    						res_k[str(1) + way] += 1
    				elif (trade_time >= tim[1]) & (trade_time < tim[2]):
    					if str(2) + way not in res_k.keys():
    						res_k[str(2) + way] = 1
    					else:
    						res_k[str(2) + way] += 1

    				elif (trade_time >= tim[2]) & (trade_time < tim[3]):
    					if str(3) + way not in res_k.keys():
    						res_k[str(3) + way] = 1
    					else:
    						res_k[str(3) + way] += 1

    				elif (trade_time >= tim[3]) & (trade_time < tim[4]):
    					if str(4) + way not in res_k.keys():
    						res_k[str(4) + way] = 1
    					else:
    						res_k[str(4) + way] += 1

    				elif (trade_time >= tim[4]) & (trade_time < tim[5]):
    					if str(5) + way not in res_k.keys():
    						res_k[str(5) + way] = 1
    					else:
    						res_k[str(5) + way] += 1

    				elif (trade_time >= tim[5]) & (trade_time < tim[6]):
    					if str(6) + way not in res_k.keys():
    						res_k[str(6) + way] = 1
    					else:
    						res_k[str(6) + way] += 1

    				elif (trade_time >= tim[6]) & (trade_time < tim[7]):
    					if str(7) + way not in res_k.keys():
    						res_k[str(7) + way] = 1
    					else:
    						res_k[str(7) + way] += 1

    				elif (trade_time >= tim[9]) & (trade_time < tim[8]) & (df.iloc[i, 3] == "SZ"): #此处关于交易市场存在问题！！！！！
    					if str(9) + way not in res_k.keys():
    						res_k[str(9) + way] = 1
    					else:
    						res_k[str(9) + way] += 1

    				else:
    					if str(8) + way not in res_k.keys():
    						res_k[str(8) + way] = 1
    					else:
    						res_k[str(8) + way] += 1
    			for m in range(9):  # 计算用户的每个交易时间偏好的频率
    				if m + 1 not in res_k.keys():
    					res_k[str(m + 1) +way] = 0.000
    				else:
    					res_k[str(m + 1) +way] = float(res_k[str(m + 1)+way]) / float(aln)
    		res = pd.DataFrame(columns=list(res_k.keys()))
    		res = res.append(res_k, ignore_index=True)
    		resal = res_al.append(res)
    	res = pd.merge(res_al[0], res_al[1], how="left", on="KHH")
    	return res
    '''

    '''
    # 品种偏好
    def get_PZPH(self, KHH):
    	datas = pd.read_excel(self.file)
    	N = ["custid", "bsflag", "market", "stktype", "matchamt","fundeffect"]
    	datas = datas[datas["matchamt"] != 0]
    	datas = datas[datas["fundeffect"]<0.00]
    	datas = datas.loc[:, N]
    	sort_label = pd.read_excel("pzph/pzbz.xlsx")                             #此处存在问题
    	datas = pd.merge(datas, sort_label, how="left", on=["market", "stktype"])      #此处存在问题
    	each_kh = {}
    	each_kh["KHH"] = int(KHH)
    	df = datas[datas['custid'] == int(KHH)]
    	dfn = df.shape[0]
    	all_money = float(df.loc[:, "matchamt"].sum())
    	if (dfn == 0) or (all_money == 0.00):
    		for i in range(17):
    			each_kh[i] = 0.00
    	else:
    		for j in range(17):  # 遍历每个品种在该用户买入的记录个数和频率
    			each_kh[j] = float(df[df["FLLB"] == j].loc[:, "matchamt"].sum()) / all_money       #此处存在问题
    	return each_kh  # 合并所有用户的结果

    '''

    # 新股情况
    def get_XGQK(self, KHH):
        datas = pd.read_excel(self.file)
        B = ["OP", "OT"]
        N = ["custid", "bsflag"]  # 提取需要的数据
        datas = datas[datas["fundeffect"] < 0.00]
        datas = datas.loc[:, N]
        each_kh = {}
        each_kh["KHH"] = int(KHH)
        df = datas[datas['custid'] == int(KHH)]
        dfn = df.shape[0]
        if dfn == 0:
            each_kh["lucky"] = -1.00
        else:
            suc = float(df[df["bsflag"] == "OP"].shape[0])  # 计算用户在两种业务科目的记录的个数
            all_times = float(df[df["bsflag"] == "OT"].shape[0])
            try:
                lucky = float(suc / all_times)  # 计算用户的幸运值
            except:
                lucky = 1
            each_kh["lucky"] = lucky
        return each_kh

    # 动量策略
    def get_DLCL(self, KHH):
        datas = pd.read_excel(self.file)
        d_kind = ts.get_industry_classified()
        d_kind["code"] = d_kind["code"].astype(int)
        datas = datas[datas["matchamt"] != 0]
        datas = pd.merge(datas, d_kind, how="left", left_on="bankcode", right_on="code", )
        datas = pd.DataFrame(datas)
        datas = datas.dropna(how='any', axis=1)
        N = ["custid", "bsflag", "market", "orderdate", "bankcode", "c_name", "fundeffect"]
        datas = datas.loc[:, N]
        res_l = []
        dataB = datas[datas["fundeffect"] < 0.00]
        dataS = datas[datas["fundeffect"] > 0.00]
        alW = [dataB, dataS]
        for data in alW:
            if data["fundeffect"].iloc[0] < 0.00:
                way = "B"
            else:
                way = "S"
            data_DK = data.loc[:, ["orderdate", "c_name"]].drop_duplicates()
            data_D = data_DK["orderdate"].drop_duplicates()
            D_n = data_D.shape[0]  # 全部的日期
            kh_res = {}
            kh_res["KHH"] = int(KHH)
            way_name = way
            for n in range(D_n):
                end_date = datetime.datetime.strptime(str(data_D.iloc[n]), '%Y%m%d')
                end_date_S = end_date.strftime('%Y-%m-%d')
                delta_D = datetime.timedelta(days=2)
                delta_W = datetime.timedelta(days=14)
                start_date_D = (end_date - delta_D).strftime('%Y-%m-%d')
                start_date_W = (end_date - delta_W).strftime('%Y-%m-%d')

                M = end_date.month
                Y = end_date.year
                end_date_M = str(Y) + "-" + str(M) + "-01"
                if M > 2:
                    start_date_M = str(Y) + "-" + str(M - 2) + "-01"
                elif M == 2:
                    start_date_M = str(Y - 1) + "-" + str(12) + "-01"
                else:
                    start_date_M = str(Y - 1) + "-" + str(11) + "-01"

                Each_D_K = data_DK[data_DK["orderdate"] == data_D.iloc[n]]  # 找出当日所有申报的股票列表

                if Each_D_K.shape[0] != 0:
                    Each_D_K_l = list(set(np.array(Each_D_K["c_name"]).tolist()))
                else:
                    continue
                for k in Each_D_K_l:
                    if str(k) == 'nan':
                        Each_D_K_l.remove(k)
                buy_D = []
                buy_W = []
                buy_M = []
                for dkl in Each_D_K_l:  # 对其中的每一个股票
                    f_kind = open("kind/" + str(dkl) + ".txt")
                    l_kind = f_kind.read().split(" ")
                    l_kind = l_kind[:-1]
                    all_r_original = pd.DataFrame()
                    lk_n = len(l_kind)
                    # 该股票池的所有股票 计算其DMW
                    for lk in l_kind:
                        lk_dic = {}
                        lk_dic["code"] = int(lk)
                        codename = int(lk)
                        filename = str(codename) + '.csv'
                        try:
                            # d_D = ts.get_k_data(str(codename), start=start_date_D, end=end_date_S, ktype="D")
                            d_D = pd.read_csv('D/' + filename)
                            d_D = d_D[d_D['date'] <= end_date_S].tail(2)

                            # d_W = ts.get_k_data(str(codename), start=start_date_W, end=end_date_S, ktype="W")
                            d_W = pd.read_csv('W/' + filename)
                            d_W = d_W[d_W['date'] <= end_date_S].tail(2)

                            # d_M = ts.get_k_data(str(codename), start=start_date_M, end=end_date_M, ktype="M")
                            d_M = pd.read_csv('M/' + filename)
                            d_M = d_M[d_M['date'] <= end_date_M].tail(2)
                            lk_dic["R_D"] = (d_D["close"].iloc[1] - d_D["close"].iloc[0]) / d_D["close"].iloc[0]
                            lk_dic["R_W"] = (d_W["close"].iloc[1] - d_W["close"].iloc[0]) / d_W["close"].iloc[0]
                            lk_dic["R_M"] = (d_M["close"].iloc[1] - d_M["close"].iloc[0]) / d_M["close"].iloc[0]
                            all_r_original = all_r_original.append(lk_dic, ignore_index=True)
                        except:
                            continue

                    if all_r_original.shape[0] == 0:
                        continue
                    else:
                        l = int(len(all_r_original) * 0.25)
                        B_D_L = np.array(
                            all_r_original.sort_values(by="R_D", ascending=False)['code'].iloc[0:l]).tolist()
                        buy_D += B_D_L
                        B_W_L = np.array(
                            all_r_original.sort_values(by="R_W", ascending=False)['code'].iloc[0:l]).tolist()
                        buy_W += B_W_L
                        B_M_L = np.array(
                            all_r_original.sort_values(by="R_M", ascending=False)['code'].iloc[0:l]).tolist()
                        buy_M += B_M_L
                Buy_D = set(buy_D)
                Buy_W = set(buy_W)
                Buy_M = set(buy_M)
                Each_kh = data[(data['custid'] == int(KHH)) & (data["orderdate"] == data_D.iloc[n])]
                bdn = 0
                bwn = 0
                bmn = 0
                bdn += Each_kh['bankcode'].isin(Buy_D).shape[0]
                bwn += Each_kh["bankcode"].isin(Buy_W).shape[0]
                bmn += Each_kh["bankcode"].isin(Buy_M).shape[0]

                if "D" + way not in kh_res.keys():
                    kh_res["D" + way] = bdn
                    kh_res["W" + way] = bwn
                    kh_res["M" + way] = bmn
                else:
                    kh_res["D" + way] += bdn
                    kh_res["W" + way] += bwn
                    kh_res["M" + way] += bmn
            number = datas[datas["custid"] == int(KHH)].shape[0]
            if number == 0:
                kh_res["D" + way] = -1.00
                kh_res["W" + way] = -1.00
                kh_res["M" + way] = -1.00
            else:
                kh_res["D" + way] = float(kh_res["D" + way]) / float(number)
                kh_res["W" + way] = float(kh_res["W" + way]) / float(number)
                kh_res["M" + way] = float(kh_res["M" + way]) / float(number)
            res = pd.DataFrame(columns=list(kh_res.keys()))
            res = res.append(kh_res, ignore_index=True)
            res_l.append(res)
        res = pd.merge(res_l[0], res_l[1], how="left", on="KHH")
        return res

    '''分时图偏好问题：无法得到确定时间的5分钟k,该函数只能得到tushare上能爬到的5分钟k数据，所以结果会有问题'''

    # 分时图偏好
    def get_FSTPH(self, KHH):
        datas = pd.read_excel(self.file)
        N = ["custid", "bsflag", "ordertime", "orderdate", "bankcode", "fundeffect"]
        datas = datas[datas["matchamt"] != 0.00]
        datas = datas.loc[:, N]
        res_l = []
        dataB = datas[datas["fundeffect"] < 0.00]
        dataS = datas[datas["fundeffect"] > 0.00]
        alW = [dataB, dataS]
        for data in alW:
            if data["fundeffect"].iloc[0] < 0.00:
                way = "B"
            else:
                way = "S"
            kh_res = {}
            kh_res["KHH"] = int(KHH)
            kh_df = data[data["custid"] == int(KHH)]
            khn = kh_df.shape[0]
            if khn == 0:
                kh_res["up" + way] = 0.00
                kh_res["down" + way] = 0.00
                kh_res["high" + way] = 0.00
                kh_res["low" + way] = 0.00
            else:
                up = 0
                down = 0
                high = 0
                low = 0
                for kn in range(khn):  # 遍历每个用户的每条记录
                    date_s = kh_df.iloc[kn, 3]
                    time_s = kh_df.iloc[kn, 2]
                    d1 = datetime.datetime.strptime(str(date_s) + " " + str(time_s)[:-2], '%Y%m%d %H%M%S')
                    d2 = d1 - datetime.timedelta(minutes=5)
                    d3 = d1 + datetime.timedelta(minutes=5)
                    end_time = d3.strftime('%Y-%m-%d %H:%M:%S')
                    start_time = d2.strftime('%Y-%m-%d %H:%M:%S')
                    df = ts.get_k_data(str(int(kh_df["ZQDM"].iloc[0])), start=start_time, end=end_time, ktype="5")
                    # df = ts.get_k_data(str(kh_df["ZQDM"].iloc[kn]), ktype="5")  # 对于该记录进行分类，判断该记录的类别
                    if df.shape[0] != 0:
                        if int(df.iloc[1, 1]) > int(df.iloc[0, 1]):
                            if int(df.iloc[1, 1]) > int(df.iloc[2, 1]):
                                high += 1
                            else:
                                up += 1
                        else:
                            if int(df.iloc[0, 1]) != 0:
                                if float((int(df.iloc[0, 1]) - int(df.iloc[1, 1])) / int(df.iloc[0, 1])) > 0.08:
                                    low += 1
                                else:
                                    down += 1
                            else:
                                down += 1
                    else:
                        continue
                kh_res["up" + way] = float(up / khn)  # 计算对应标签的频率
                kh_res["down" + way] = float(down / khn)
                kh_res["high" + way] = float(high / khn)
                kh_res["low" + way] = float(low / khn)
            res_E = pd.DataFrame(columns=list(kh_res.keys()))
            res_E = res_E.append(kh_res, ignore_index=True)
            res_l.append(res_E)
        res = pd.merge(res_l[0], res_l[1], how="left", on="KHH")
        return res

    # 新股偏好上分位数
    def get_XG_sta(self, datas):
        if self.X_flag == 0:
            res_E = pd.DataFrame(columns=['KHH', 'XGPH'])
            khh = self.get_KHH(self.file)
            for i, kh in enumerate(khh):
                kh_res = {}
                kh_res["KHH"] = kh
                df = datas[datas["custid"] == kh]
                if df.shape[0] != 0:
                    kh_pre = float(float(df[df["bsflag"] == "OP"].shape[0]) / float(df.shape[0]))
                else:
                    continue
                kh_res["XGPH"] = kh_pre
                res_E.loc[i] = kh_res
            # res_E.append(kh_res, ignore_index=True)
            # print res_E.head()
            res_B = res_E
            mean = res_B["XGPH"].mean()
            std = res_B["XGPH"].std()
            res_B["XGPH"] = np.asarray(((np.asarray(res_B["XGPH"]) - mean) / std), dtype=np.float)
            key_n = int(res_B.shape[0] * 0.2)
            res = res_B.sort_values(by="XGPH", ascending=False)
            key = res.iloc[key_n, 0]
            self.X_sta = res_E[res_E["KHH"] == int(key)].loc[:, "XGPH"].iloc[0]
            self.X_flag = 1
        return self.X_sta

    # 新股偏好
    def get_XGPH(self, KHH):
        datas = pd.read_excel(self.file)
        N = ["custid", "bsflag"]
        datas = datas[datas["matchamt"] != 0.00]
        datas = datas[datas["fundeffect"] < 0.00]
        datas = datas.loc[:, N]
        kh_res = {}
        kh_res["KHH"] = int(KHH)
        df = datas[datas["custid"] == int(KHH)]
        if df.shape[0] == 0:
            kh_pre = -1.00
        else:
            kh_pre = float(float(df[df["bsflag"] == "OP"].shape[0]) / float(df.shape[0]))
        kh_res["XGPH"] = kh_pre
        sta = max(self.get_XG_sta(datas), 0.01)
        if kh_res["XGPH"] != -1.00:
            if kh_res["XGPH"] > sta:
                kh_res["XGPH"] = 1
            else:
                kh_res["XGPH"] = 0
        else:
            kh_res["XGPH"] = -1
        return kh_res

    '''
    # B股偏好
    def get_BGPHKHH(self, KHH):
    	datas = pd.read_excel(self.file)
    	N = ["custid", "bsflag", "matchamt", "stktype", "moneytype"]
    	datas = datas[datas["fundeffect"]<0.00]
    	datas = datas.loc[:, N]
    	datas = datas[datas["matchamt"] != 0]
    	each_kh = {}
    	each_kh["KHH"] = int(KHH)
    	df = datas[datas['custid'] == int(KHH)]
    	dfn = df.shape[0]
    	all_money = df.loc[:, "matchamt"].sum()
    	if (dfn == 0) & (all_money == 0.00):
    		each_kh["FB"] = -1.00
    		each_kh["B"] = -1.00
    	else:
    		dc = df[df["stktype"] == "B0"]                                            #存在问题！！！！！
    		each_kh["B"] = float(float(dc.loc[:, "matchamt"].sum()) / float(all_money))
    		each_kh["FB"] = 1 - each_kh["B"]
    	return each_kh

    '''
    '''
    # B股情况
    def get_BGQK(self, KHH):
    	datas = pd.read_excel(self.file)
    	N = ["custid", "bsflag", "matchamt", "stktype", "moneytype"]
    	datas = datas[datas["fundeffect"]<0.00]
    	datas = datas.loc[:, N]
    	datas = datas[datas["matchamt"] != 0]
    	each_kh = {}
    	each_kh["KHH"] = int(KHH)
    	df = datas[datas['custid'] == int(KHH)]
    	all_money = float(df.loc[:, "matchamt"].sum())
    	df = df[df["stktype"] == "B0"]                                         #存在问题！！！！！
    	dfn = df.shape[0]
    	all_B_money = float(df.loc[:, "matchamt"].sum())
    	if (dfn == 0) & (all_B_money == 0.00):
    		each_kh["FB"] = 1.00
    		each_kh["HB"] = 0.00
    		each_kh["DB"] = 0.00
    	else:
    		each_kh["FB"] = float(float(df[df["stktype"] != "B0"].loc[:, "matchamt"].sum()) / float(all_money))  #存在问题！！！！！
    		each_kh["HB"] = float(float(df[df["moneytype"] == "HKD"].loc[:, "matchamt"].sum()) / float(all_B_money))        #存在问题！！！！！
    		each_kh["DB"] = float(1 - each_kh["HB"]) 
    	return each_kh

    '''

    # ST股上分位数
    def get_ST_sta(self, datas):
        if self.S_flag == 0:
            khh = self.get_KHH(self.file)
            res_E = pd.DataFrame(columns=["KHH", "ST"])
            for j, kh in enumerate(khh):
                each_kh = {}
                each_kh["KHH"] = int(kh)
                df = datas[datas['custid'] == int(kh)]
                all_money = float(df.loc[:, "matchamt"].sum())
                dfn = df.shape[0]
                if (dfn != 0) & (all_money != 0.00):
                    stn = 0.00
                    for i in range(dfn):
                        if u'ST' in df.iloc[i, 3]:
                            stn += df.iloc[i, 2]
                    each_kh["ST"] = float(stn / all_money)
                else:
                    each_kh["ST"] = 0.00
                res_E = res_E.append(each_kh, ignore_index=True)
            res_B = res_E
            mean = res_B["ST"].mean()
            std = res_B["ST"].std()
            res_B["ST"] = np.asarray(((np.asarray(res_B["ST"]) - mean) / std), dtype=np.float)
            key_n = int(res_B.shape[0] * 0.2)
            res = res_B.sort_values(by="ST", ascending=False)
            key = res.iloc[key_n, 0]
            self.S_sta = res_E[res_E["KHH"] == int(key)].loc[:, "ST"].iloc[0]
            self.S_flag = 1
        return self.S_sta

    # ST股偏好
    def get_STPH(self, KHH):
        datas = pd.read_excel(self.file)
        N = ["custid", "bsflag", "matchamt", "stkname"]
        datas = datas[datas["fundeffect"] < 0.00]
        datas = datas.loc[:, N]
        datas = datas[datas["matchamt"] != 0]
        each_kh = {}
        each_kh["KHH"] = int(KHH)
        df = datas[datas['custid'] == int(KHH)]
        all_money = float(df.loc[:, "matchamt"].sum())
        dfn = df.shape[0]
        if (dfn == 0) & (all_money == 0.00):
            each_kh["ST"] = -1.00
        else:
            stn = 0.00
            for i in range(dfn):
                if u'ST' in df.iloc[i, 3]:
                    stn += df.iloc[i, 2]
            each_kh["ST"] = float(stn / all_money)
        sta = max(self.get_ST_sta(datas), 0.01)
        if each_kh["ST"] != -1.00:
            if each_kh["ST"] > sta:
                each_kh["ST"] = 1
            else:
                each_kh["ST"] = 0
        else:
            each_kh["ST"] = -1
        return each_kh

    # 次新股上分位数
    def get_CX_sta(self, basic_stk, datas):
        if self.C_flag == 0:
            khh = self.get_KHH(self.file)
            res_E = pd.DataFrame(columns=["KHH", "CX"])
            for i, kh in enumerate(khh):
                each_kh = {}
                each_kh["KHH"] = int(kh)
                df = datas[datas['custid'] == int(kh)]
                all_money = float(df.loc[:, "matchamt"].sum())
                dfn = df.shape[0]
                if (dfn != 0) & (all_money != 0.00):
                    for n in range(dfn):
                        CX_money = 0.00
                        code = df.iloc[n, 3]
                        buy_year = int(str(df.iloc[n, 4])[:4])
                        mkt_year = int(str(basic_stk[basic_stk["code"] == code].iloc[0, 1])[:4])
                        if buy_year - mkt_year > 1:
                            CX_money += float(df[df["bankcode"] == code].loc[:, "matchamt"])
                        else:
                            continue
                    each_kh["CX"] = float(CX_money / all_money)
                else:
                    each_kh["CX"] = 0.00
                res_E = res_E.append(each_kh, ignore_index=True)

            res_B = res_E
            mean = res_B["CX"].mean()
            std = res_B["CX"].std()
            res_B["CX"] = np.asarray(((np.asarray(res_B["CX"]) - mean) / std), dtype=np.float)
            key_n = int(res_B.shape[0] * 0.2)
            res = res_B.sort_values(by="CX", ascending=False)
            key = res.iloc[key_n, 0]
            self.C_sta = res_E[res_E["KHH"] == int(key)].loc[:, "CX"].iloc[0]
            self.C_flag = 1
        return self.C_sta

    # 次新股偏好
    def get_CXPH(self, KHH):
        datas = pd.read_excel(self.file)
        basic_stk = pd.read_csv("basic.csv")
        basic_stk = basic_stk.loc[:, ["code", "timeToMarket"]]
        # stocks_f = open(u"次新股.txt")
        # stocks = stocks_f.read().split(" ")[:-1]
        for j in range(len(stocks)):
            stocks[j] = int(stocks[j])
        N = ["custid", "bsflag", "matchamt", "bankcode", "orderdate"]
        datas = datas[datas["fundeffect"] < 0.00]
        datas = datas.loc[:, N]
        datas = datas[datas["matchamt"] != 0]
        each_kh = {}
        each_kh["KHH"] = int(KHH)
        df = datas[datas['custid'] == int(KHH)]
        all_money = float(df.loc[:, "matchamt"].sum())
        dfn = df.shape[0]
        if (dfn != 0) & (all_money != 0.00):
            each_kh["CX"] = -1.00
        else:
            for n in range(dfn):
                CX_money = 0.00
                code = df.iloc[n, 3]
                buy_year = int(str(df.iloc[n, 4])[:4])
                mkt_year = int(str(basic_stk[basic_stk["code"] == code].iloc[0, 1])[:4])
                if buy_year - mkt_year > 1:
                    CX_money += float(df[df["bankcode"] == code].loc[:, "matchamt"])
                else:
                    continue

            each_kh["CX"] = float(CX_money / all_money)
        sta = max(self.get_CX_sta(basic_stk, datas), 0.01)
        if each_kh["CX"] != -1.00:
            if each_kh["CX"] > sta:
                each_kh["CX"] = 1
            else:
                each_kh["CX"] = 0
        else:
            each_kh["CX"] = -1
        return each_kh

    # 可转债上分位数
    def get_KZZ_sta(self, datas):
        s = u'转债'
        if self.K_flag == 0:
            khh = self.get_KHH(self.file)
            res_E = pd.DataFrame()
            for j, kh in enumerate(khh):
                each_kh = {}
                df = datas[datas['custid'] == int(kh)]
                all_money = float(df.loc[:, "matchamt"].sum())
                dfn = df.shape[0]
                if (dfn != 0) & (all_money != 0.00):
                    stn = 0.00
                    each_kh["KHH"] = int(kh)
                    for i in range(dfn):
                        if s in df.iloc[i, 3]:
                            stn += df.iloc[i, 2]

                    each_kh["KZZ"] = float(stn / all_money)
                    res_E = res_E.append(each_kh, ignore_index=True)

            res_B = res_E[res_E["KZZ"] != -1.00]
            mean = res_B["KZZ"].mean()
            std = max(res_B["KZZ"].std(), 0.01)
            res_B["KZZ"] = np.asarray(((np.asarray(res_B["KZZ"]) - mean) / std), dtype=np.float)
            key_n = int(res_B.shape[0] * 0.2)
            res = res_B.sort_values(by="KZZ", ascending=False)
            key = res.iloc[key_n, 0]
            self.K_sta = res_E[res_E["KHH"] == int(key)].loc[:, "KZZ"].iloc[0]
            self.K_flag = 1
        return self.K_sta

    # 可转债偏好
    def get_KZPH(self, KHH):
        datas = pd.read_excel(self.file)
        N = ["custid", "bsflag", "matchamt", "stkname"]
        s = u'转债'
        datas = datas[datas["fundeffect"] < 0.00]
        datas = datas.loc[:, N]
        datas = datas[datas["matchamt"] != 0]
        each_kh = {}
        each_kh["KHH"] = int(KHH)
        df = datas[datas['custid'] == int(KHH)]
        all_money = float(df.loc[:, "matchamt"].sum())
        dfn = df.shape[0]
        if (dfn == 0) & (all_money == 0.00):
            each_kh["KZZ"] = -1.00
        else:
            stn = 0.00
            for i in range(dfn):
                if s in df.iloc[i, 3]:
                    stn += df.iloc[i, 2]
            each_kh["KZZ"] = float(stn / all_money)

        sta = self.get_KZZ_sta(datas)
        if each_kh["KZZ"] != -1.00:
            if each_kh["KZZ"] > sta:
                each_kh["KZZ"] = 1
            else:
                each_kh["KZZ"] = 0
        else:
            each_kh["KZZ"] = -1
        return each_kh

    def tor_pref(self, khh):
        return invest_pref(self.file, khh, 'daily', 'tor', 3)

    def fund_pref(self, khh):
        return invest_pref(self.file, khh, 'quarterly', 'fund', 2)

    def otstd_pref(self, khh):
        return invest_pref(self.file, khh, 'static', 'outstanding', 3)

    def op_pref(self, khh):
        return invest_pref(self.file, khh, 'daily', 'op', 3)

    def pe_pref(self, khh):
        return invest_pref(self.file, khh, 'static', 'pe', 3)

    def pb_pref(self, khh):
        return invest_pref(self.file, khh, 'static', 'pb', 3)

    def npr_pref(self, khh):
        return invest_pref(self.file, khh, 'static', 'npr', 2)

    def top_pref(self, khh):
        return invest_pref(self.file, khh, 'daily', 'top', 2)

    def macd_pref(self, khh):
        return operate_pref(self.file, khh, 'MACD')

    def kdj_pref(self, khh):
        return operate_pref(self.file, khh, 'KDJ')

    def industry_pref(self, khh):
        return invest_pref(self.file, khh, 'static', 'industry', 49)

    def concept_pref(self, khh):
        return invest_pref(self.file, khh, 'static', 'concept', 99)

if __name__ == '__main__':
    x = user_label()
    for khh in x.khhs:
        cp = x.concept_pref(khh)
        idx = np.nonzero(cp[khh])
        print(cp)
        print(idx)