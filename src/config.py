

FINAL_FEATURES =[
       'is_acq_merchant_country_equal', 'pos_Entry_Mode', 'pos_Condition_Code',
       'transaction_type', 'card_present', 
       'merchant_trans_outlier', 'merchant_code_fraud_ratio',
       'merchant_name_fraud_ratio', 
       'mean_cus_card_trans_gap', 'credit_used_ratio', 'credit_left_ratio',
       'balance_ratio',
       'cus_trans_ratio_by_cat', 
       'cus_trans_ratio_by_merchant', 'is_correct_CVV', 'transaction_hour',
       'transaction_day_of_week', 'account_months','trans_gap_ratio_cus','trans_gap_ratio_card']

ID = ['accountNumber','customerId']

TARGET = 'isFraud'