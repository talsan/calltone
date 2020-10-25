from utils_athena import query

query_parameters = dict(
    return_df=False,  # return a pandas data frame? (if False: returns s3 key)
    download_path='./extracts/returns_extract_rundate_20200930.csv',  # where to save locally (if None: no local download)
    output_bucket='fool-calls-athena-output',  # bucket for query output (bucket needs to exist)
    work_group='qc'
)

s3_output_location = query(sql_string='SELECT * '
                                      'FROM av_prices ' 
                                      'WHERE rundate = date(\'2020-09-30\' )'
                                      'AND ticker IN '
                                      '(SELECT DISTINCT ticker '
                                      'FROM ishares_holdings '
                                      'WHERE etf = \'IWB\')', ** query_parameters)
