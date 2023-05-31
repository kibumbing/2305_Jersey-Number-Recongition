import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory="")
# mySoccerNetDownloader.downloadDataTask(task="jersey-2023", split=["train","test","challenge"])

mySoccerNetDownloader.downloadDataTask(task="tracking", split=["train", "test", "challenge"])

mySoccerNetDownloader.downloadDataTask(task="tracking-2023", split=["train", "test", "challenge"])