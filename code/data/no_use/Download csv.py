import csv
import requests

CSV_URL = 'http://samplecsvs.s3.amazonaws.com/Sacramentorealestatetransactions.csv'


with requests.Session() as s:
    download = s.get(CSV_URL)

    decoded_content = download.content.decode('utf-8')

    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
    my_list = list(cr)
    for row in my_list:
        print(row)
        
