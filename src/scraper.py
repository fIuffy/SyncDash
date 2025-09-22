import requests

def download_level(level_id):
    url = "https://www.boomlings.com/database/downloadGJLevel22.php"
    data = ""
    headers = {'User-Agent': ''}
    response = requests.post(url,data=data,headers=headers)
    
    if response.status_code == 200:
        level_data = response.content
        return level_data
    else:
        raise Exception(f"Failed to download level. Status code: {response.status_code}")

print(download_level("138"))