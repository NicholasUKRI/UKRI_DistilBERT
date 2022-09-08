import requests
import json
import time
import pydash as _
from retrying import retry
import os

# create data folder for data
if not os.path.exists('data'):
    os.makedirs('data')

list_projects = []

# this will pull every project, dating back to 1991. this could be refined to only pull from a more suitable year i.e 10 years back
@retry(wait_random_min=60, wait_random_max=120, stop_max_attempt_number=10)
def fetch_projects():
    new_results = True
    page = 1
    while new_results:
        try:
            url = "https://gtr.ukri.org/project"
            params = {"page": page,
                      "fetchSize": 100}
            projects = requests.get(url,
                                    params=params,
                                    headers={"accept": "application/json"}).json()['projectsBean']['project']

            for project in projects:
                list_projects.append({'ref': project.get('grantReference', ''),
                                      'title': project.get('title', ''),
                                      'funder': _.get(project, 'fund.funder.name', ''),
                                      'category': project.get('grantCategory', ''),
                                      'abstract': project.get('abstractText', ''),
                                      'funding': _.get(project, 'fund.valuePounds', ''),
                                      'start': _.get(project, 'fund.start', ''),
                                      'end': _.get(project, 'fund.end', '')})

            page += 1
            time.sleep(4)
            print(page)
        except ValueError:
            break

    list_projects = {str(index): lst for index, lst in enumerate(list_projects)}
    filepath = "data\\projects.json" if dir is not None else "projects.json"
    with open(filepath, 'w') as f:
        json.dump(list_projects, f)

    return list_projects


fetch_projects()



