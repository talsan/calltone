import boto3
import time
from config import Aws
from botocore.exceptions import ClientError
import pandas as pd
from io import BytesIO
from datetime import datetime
import pprint
import logging

log = logging.getLogger(__name__)

aws_session = boto3.Session(aws_access_key_id=Aws.AWS_KEY,
                            aws_secret_access_key=Aws.AWS_SECRET)

'''
Overview: 
    The sole purpose of this module is to expose a single function: "query" (or "athena_helpers.query", when imported).

Intended Usage:
    As a module (i.e imported so the user can use athena to query underlying s3 objects)

    Example: 
        df_big = athena_helpers.query("select * from qcdb.etf_holdings")     

Input Parameters: 
    sql_str -- ETF ticker you wish to download
    database -- database context (defaults to "qcdb")
    cleanup -- whether to delete s3 files after the dataframe is returned (not yet implemented)

Output: 
    pandas dataframe with etf holdings
'''


class AthenaQuery():

    def __init__(self, user_query, output_bucket, region, database, work_group, sleep_between_requests, query_timeout):
        self.query = user_query
        self.database = database
        self.work_group = work_group
        self.sleep_between_requests = sleep_between_requests
        self.query_timeout = query_timeout
        self.query_output_bucket = output_bucket
        self.region = region
        self.athena_client = aws_session.client('athena', region_name=region)
        self.s3_client = aws_session.client('s3', region_name=region)
        self.s3_object_html_root = 'https://s3.console.aws.amazon.com/s3/object'
        self.execution_id = ''  # provided by aws after query is submitted
        self.output_dir = ''  # derived from execution_id (to keep query and output closely linked)
        self.output_df = pd.DataFrame()

    def init_query(self):
        run_time = datetime.now()
        self.output_loc = f'{run_time.year}/{run_time.month}/{run_time.day}'
        response = self.athena_client.start_query_execution(
            QueryString=self.query,
            QueryExecutionContext={
                'Database': self.database,
            },
            ResultConfiguration={
                'OutputLocation': f's3://{self.query_output_bucket}/{self.output_loc}'
            },
            WorkGroup=self.work_group,
        )
        return response

    def run_query(self):
        query_execution = self.init_query()
        self.execution_id = query_execution['QueryExecutionId']
        state = 'QUEUED'

        start_time = datetime.now()
        while ((datetime.now() - start_time).total_seconds() < self.query_timeout) \
                and (state in ['RUNNING', 'QUEUED']):
            response = self.athena_client.get_query_execution(QueryExecutionId=self.execution_id)

            if 'QueryExecution' in response and \
                    'Status' in response['QueryExecution'] and \
                    'State' in response['QueryExecution']['Status']:
                state = response['QueryExecution']['Status']['State']

                if state == 'FAILED':
                    print(response['QueryExecution']['Status']['StateChangeReason'])
                    log.error(pprint.pformat(response))
                    return False
                elif state == 'SUCCEEDED':
                    s3_path = response['QueryExecution']['ResultConfiguration']['OutputLocation']
                    s3_key = s3_path.replace(f's3://{self.query_output_bucket}/', '')
                    log.info(pprint.pformat(response))
                    return s3_key

            time.sleep(self.sleep_between_requests)

        return False

    def get_query_output(self, s3_key):
        try:
            obj = self.s3_client.get_object(Bucket=self.query_output_bucket,
                                            Key=s3_key)
        except ClientError as e:
            log.error(f'{e.response["Error"]["Code"]}\n'
                      f'cant find file {s3_key}')
            raise
        df = pd.read_csv(BytesIO(obj['Body'].read()), dtype=str)
        return df

    def display_s3_url(self):
        s3_output_url = f'{self.s3_object_html_root}/{self.query_output_bucket}/{self.output_loc}/{self.execution_id}.csv'
        s3_output_str = f's3 output: {s3_output_url}'
        log.info(s3_output_str)
        print(s3_output_str)

    def display_athena_url(self):
        athena_query_url = f'https://{self.region}.console.aws.amazon.com/athena/home?' \
                           f'region={self.region}#query/history/{self.execution_id}'
        athena_query_str = f'athena query: {athena_query_url}'
        log.info(athena_query_str)
        print(athena_query_str)


def query(sql_string, return_df=True, download_path=None, output_bucket='athena-query-outputs', region='us-west-2', database='qcdb',
          work_group='primary', sleep_between_requests=3, query_timeout = 600, cleanup=False):

    aq = AthenaQuery(sql_string, output_bucket, region, database, work_group, sleep_between_requests, query_timeout)
    s3_key = aq.run_query()

    # display links for aws consoles for s3 object and athena queries
    aq.display_s3_url()
    aq.display_athena_url()

    # future logic here to remove s3 query output data (will be done periodically for now)
    if cleanup:
        pass

    if download_path is not None:
        aq.s3_client.download_file(aq.query_output_bucket, s3_key, download_path)

    if return_df:
        if download_path is not None:
            df = pd.read_csv(download_path, dtype=str)
        else:
            df = aq.get_query_output(s3_key)
        return df
    else:
        return s3_key