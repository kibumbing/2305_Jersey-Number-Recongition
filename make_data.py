import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory="")
mySoccerNetDownloader.downloadDataTask(task="jersey-2023", split=["train","test","challenge"])