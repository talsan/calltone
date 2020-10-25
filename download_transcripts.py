from utils_athena import query

query_parameters = dict(
    return_df=False, # return a pandas data frame? (if False: returns s3 key)
    download_path='./extracts/foolcalls_extract_20201001.csv', # where to save locally (if None: no local download)
    output_bucket='fool-calls-athena-output', # bucket for query output (bucket needs to exist)
    work_group='qc'
)

s3_output_location = query(sql_string='SELECT '
                                      'idx.cid as cid,'
                                      'call_url,ticker,company_name,'
                                      'publication_time_published,publication_time_updated,period_end,'
                                      'fiscal_period_year,fiscal_period_qtr,call_date,duration_minutes,'
                                      'statement_num,section,statement_type,role,text'
                                      ' FROM fool_call_index idx '
                                      'JOIN fool_call_statements st '
                                      'ON idx.cid = st.cid ',
                                      # 'where ticker = \'AAPL\'',
                           **query_parameters)
