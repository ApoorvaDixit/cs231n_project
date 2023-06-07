
import time
import pytz
from datetime import datetime

def get_timestr():
    t0 = time.time()
    pst = pytz.timezone('America/Los_Angeles')
    datetime_pst = datetime.fromtimestamp(t0, pst)
    return datetime_pst.strftime("%I:%M %p, %B %d")