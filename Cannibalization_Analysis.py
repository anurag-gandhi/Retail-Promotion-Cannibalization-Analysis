
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


# In[ ]:

#read input files
date_map = pd.read_csv('C:/Users/Anurag.Gandhi/Desktop/date_map.csv')
promotions = pd.read_csv('C:/Users/Anurag.Gandhi/Desktop/hnp_promotions.csv')
#read transaction data
txn_data = pd.read_csv('C:/Users/Anurag.Gandhi/Desktop/txn_data_2_years.csv'
                  ,encoding = "ISO-8859-1")


# In[ ]:

txn_data['NEW_DATE'] = pd.to_datetime(txn_data['TRANSACTION_DATE'], format = '%d%b%Y')
date_map['NEW_DATE'] = pd.to_datetime(date_map['CALENDAR_DATE'], format = '%m/%d/%Y')


# In[ ]:

txn_data = pd.merge(txn_data, date_map, how = 'left', on = 'NEW_DATE', sort = False)


# In[ ]:

txn_data.rename(columns = {'WEEK_NO': 'PROM_WEEK_NO'}, inplace = True)
promotions.rename(columns = {'sku': 'SKU', 'prom_week_no': 'PROM_WEEK_NO'}, inplace = True)


# In[ ]:

txn_data = pd.merge(txn_data, promotions[promotions['zone_id'] == 5][['SKU', 'PROM_WEEK_NO', 'promotions', 'promotions_NC']], 
                 how = 'left', on = ['SKU', 'PROM_WEEK_NO'], sort = False)


# In[ ]:

txn_data[['promotions','promotions_NC']] = txn_data[['promotions','promotions_NC']].fillna(0)


# In[ ]:

#Unique ID for Customer-Day (proxy for transaction)
txn_data['UNIQUE_CUST_TRANS'] = txn_data['INFERRED_CUSTOMER_ID'].map(str) + txn_data['TRANSACTION_DATE']


# In[ ]:

#read correlation output
sisters_list = pd.read_csv('H:/Depository/Sirsa/Promotion_Analytics/Research/Anurag/Basket/sisters_corr_hnp_v3.csv',encoding = "ISO-8859-1")

#business inputs 
#sisters_list = pd.DataFrame(sisters_list[sisters_list['buyer_id_2'].isin([4494,4266,4502])]).reset_index


# In[ ]:

sisters_list_corr = sisters_list

    


# ## Lift (X->Y) = confidence(X->Y) / support(Y)
# 

# In[ ]:

trans_sku = txn_data[['UNIQUE_CUST_TRANS','SKU','PROM_WEEK_NO']].drop_duplicates()
#loop for each combination
start_time = time.time()
for i in range(0,len(sisters_list)):
#for i in range(0,363):
    #SKU A: Dishwash SKU
    SKU_A = sisters_list.ix[i,'SKU_1']
    
    #SKU B: Sister SKU
    SKU_B = sisters_list.ix[i,'SKU_2']
    
    #filter weeks where only ONE of the products was on promotion
    test = txn_data[txn_data['SKU'].isin([SKU_A,SKU_B])][['PROM_WEEK_NO', 'promotions', 'SKU']].drop_duplicates()
    test_2 = test.pivot(index = 'PROM_WEEK_NO', columns = 'SKU', values = 'promotions').fillna(0)
    test_2['sum'] = test_2[SKU_A] + test_2[SKU_B]
    test_2['product'] = test_2[SKU_A] * test_2[SKU_B]
    prom_week_list = test_2[(test_2['sum'] > 0) & (test_2['product'] == 0)].index.values
    
    #Customers who bought these two SKUs:
    hnp_pair = txn_data[txn_data['SKU'].isin([SKU_A,SKU_B])][['INFERRED_CUSTOMER_ID', 'SKU']].drop_duplicates()
    sku_count_by_customer = hnp_pair.groupby( ['INFERRED_CUSTOMER_ID'], sort = False).count()
    
    #ATLEAST 1 customer should have bought both the products (this removes unrelated products)
    if ((max(sku_count_by_customer['SKU']) > 1) & (len(prom_week_list) > 0 )):
        
        #distinct SKU transactions
        trans_sku_pair = trans_sku[trans_sku['PROM_WEEK_NO'].isin(prom_week_list)][['UNIQUE_CUST_TRANS','SKU']]
        trans_sku_pair = trans_sku_pair[trans_sku_pair['SKU'].isin([SKU_A,SKU_B])].drop_duplicates()
        pivoted = trans_sku_pair.groupby( ['UNIQUE_CUST_TRANS'], sort = False).count()
        
        #Support (A) = number/fraction of transactions with A
        support_a = trans_sku_pair[trans_sku_pair['SKU'] == SKU_A].shape[0]
        
        #Suport (B) = number/fraction of transactions with B
        support_b = trans_sku_pair[trans_sku_pair['SKU'] == SKU_B ].shape[0]
        
        #Support (A, B) = number/fraction of transactions with both A & B
        intersection = pivoted[pivoted['SKU'] == 2].shape[0]
        sisters_list.ix[i,'SUPPORT A'] = support_a
        sisters_list.ix[i,'SUPPORT B'] = support_b
        sisters_list.ix[i,'SUPPORT A_B'] = intersection
        
        #Confidence = Support (A, B) / Support (A)
        sisters_list.ix[i,'CONFIDENCE'] = np.float64(intersection)/support_a
        
        #Lift = Support (A, B) / ( Support (A) * Support (B) )
        sisters_list.ix[i,'LIFT'] = np.float64(intersection)/(support_a*support_b)
    else:
        sisters_list.ix[i,'LIFT'] = 1000
    print (i + 1, " of", len(sisters_list), " done.")
end_time = time.time()
print ("Elapsed Time:",(end_time - start_time))


# In[ ]:

sisters_list


# In[ ]:

sisters_list[sisters_list['sub_category_2'] == 417]


# In[ ]:

results = sisters_list[['SKU_1','sku_desc_1', 'sub_category_name_1', 'SKU_2', 'sku_desc_2', 'sub_category_name_2', 
                        'SUPPORT A', 'SUPPORT B', 'SUPPORT A_B', 'CONFIDENCE', 'LIFT', 'count_stores', 'base', 'avg_corr']]


# In[ ]:

sisters_list.columns


# In[ ]:

results[(results['LIFT'] < 3e-7) ]


# In[ ]:

results_0_lift = results[ (results['LIFT'] < 1e-6) & (results['SUPPORT A'] >= 300) & (results['SUPPORT B'] >=300) ]
results_0_lift['SKU_1'].drop_duplicates().shape


# In[ ]:

results_0_lift.to_csv('results_dishwash_3.csv')


# In[ ]:

results_0_lift


# In[ ]:

hnp


# In[ ]:

SKU_A = 7512416
SKU_B = 2947365
hnp_pair = hnp[hnp['SKU'].isin([SKU_A,SKU_B])][['INFERRED_CUSTOMER_ID', 'SKU']].drop_duplicates()
sku_count_by_customer = hnp_pair.groupby( ['INFERRED_CUSTOMER_ID']).count()


# In[ ]:

customer_list = sku_count_by_customer[sku_count_by_customer['SKU'] == 2].index.values


# In[ ]:

data_pair = hnp[hnp['SKU'].isin([SKU_A,SKU_B]) & (hnp['INFERRED_CUSTOMER_ID'].isin(customer_list))]


# In[ ]:

data_pair


# In[ ]:

SKU_list = [7512240, 2947365, 7495079, 7512244, 7512320, 7512370, 7512427, 7624019]
txn_test = txn_data[txn_data['SKU'].isin(SKU_list)]


# In[ ]:

def classify_sisters(row):
    if row['SKU'] == 7512240:
        return 'Parent'
    else:
        return 'Sister'

txn_test['Status'] = txn_test.apply(classify_sisters, axis = 1)


# In[ ]:

txn_test


# In[ ]:

txn_test.to_csv('boundary_transactions.csv')


# In[ ]:

txn_data[(txn_data['INFERRED_CUSTOMER_ID'] == 40000977860936) & (txn_data['SKU'] == 7512242)]


# In[ ]:

sisters_list[(sisters_list['SKU_1'] == 7512240) & (sisters_list['SKU_2'] == 7512242)]


# In[ ]:



