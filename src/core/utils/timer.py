import time


class Timer(object):
    """
    Computes elapsed time.
    
    TODO remove all prints in this class and return the messages instead.
    """
    def __init__(self, timer_name):

        self.timer_name = timer_name

        self.is_running = True

        # the total time elapsed when the timer is running
        self.total_time = 0

        # the moment the timer was created
        self.init_moment = time.time()

        # the moment the timer was started
        self.start_moment = time.time()

        # the moment controled by the `get_interval` method
        self.interval_moment = time.time()

        print("<> <> <> Starting Timer [{}] <> <> <>".format(self.timer_name))

    def update_total_time(self, return_full_float=False):
        """
        Update the total time elapsed and return it.

        `return_full_float`: if True, return the total time with full precision.
        """
        if self.is_running:
            self.total_time += time.time() - self.start_moment
        else:
            print("*** Warning: This timer is already finished, so it can't be updated! ***")

        if return_full_float:
            return self.total_time
        else:
            return round(self.total_time, 2)
        
    def get_interval(self, interval_name=''):
        """
        Get the interval time from the last interval moment to the current moment.

        This method also resets the current interval moment.

        TODO don't reset the interval moment & add a method to reset it.
        """
        interval_time_hms = self._sec_to_hms(round(time.time() - self.interval_moment, 2))
        msg = "<> <> Timer [{}] <> <> Interval [{}]: {} <> <>".format(self.timer_name, 
                                                                interval_name, interval_time_hms)
        print(msg)
        self.interval_moment = time.time()           # reset the interval moment
        return msg

    def reset(self):
        """
        Reset all the timer's attributes except the timer name.
        """
        self.is_running = True
        self.total_time = 0
        self.init_moment = time.time()
        self.start_moment = time.time()
        self.interval_moment = time.time()
        return self

    def pause(self):
        if self.is_running:
            self.update_total_time()

            # set these attributes to None to indicate that the timer is paused
            self.start_moment = None
            self.interval_moment = None
            
            self.is_running = False
        return self

    def resume(self):
        if not self.is_running:
            self.start_moment = time.time()
            self.interval_moment = time.time()
            self.is_running = True
        else:
            print("*** Warning: This timer is already running! ***")
        return self

    def finish(self):
        if self.is_running:
            total_time_hms = self._sec_to_hms(self.update_total_time(return_full_float=True))
            self.is_running = False
        else:
            print("*** Warning: This timer is already finished! ***")
        
        print("<> <> <> Finished Timer [{}] <> <> <> Total time elapsed: {} <> <> <>".format(
                                                                self.timer_name, total_time_hms))

    def _sec_to_hms(self, seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return "%02dh %02dm %02ds" % (h, m, s)
