
class MovingThreshold:
    def __init__(self, config, **kw):
        self.amortisation = config['shift_threshold_ball'] 
        self.greedy_shift_time = config['shift_threshold_time'] 
        self.min_en = kw.get('target_energy') 
        self.success_thresh = config["success_thresh"]
        self.succ_radius_shift = config["succ_radius_shift"]
        self.succes_switch = config["succes_switch"]
        self.current_threshold = config['accept_err']

        self.lowest_energy = self.min_en + self.current_threshold
        self.success_counter = 0
        self.radius_shift_counter = 0
        self.call_counter = 0
        

    def reduce_amortisation(self):
        if self.success_thresh:
            """ Reduce amortisation radius after some amount of succeses"""
            self.success_counter += 1
            if self.success_counter >= self.success_thresh and self.radius_shift_counter < self.succ_radius_shift and self.succes_switch > abs(self.min_en - self.lowest_energy):
                self.current_threshold -= self.amortisation/self.succ_radius_shift
                self.success_counter = 0
                self.radius_shift_counter += 1
        
        return self.current_threshold


    def greedy_shift(self):
        self.call_counter += 1            
        if self.call_counter > 10 and (self.call_counter%(self.greedy_shift_time)) == 0:
            if self.amortisation:
                self.current_threshold = abs(self.min_en - self.lowest_energy) + self.amortisation

                if self.success_thresh:
                    self.radius_shift_counter = 0
                    self.success_counter = 0
            else:
                self.current_threshold = abs(self.min_en - self.lowest_energy)
  
        return self.current_threshold
    
    def get_current_threshold(self):
        return self.current_threshold

    def update_threshold(self, **kw):
        energy_done = kw.get('energy_done')
        if energy_done:
            self.reduce_amortisation()
        self.greedy_shift()
    
class SuccesCountThreshold:
    def __init__(self, config, **kw):
        self.min_en = kw.get('target_energy') 
        self.success_thresh = config["success_thresh"]
        self.current_threshold = config['accept_err']

        self.lowest_energy = self.min_en + self.current_threshold
        self.success_counter = 0
  
    def greedy_shift(self):
        if self.success_thresh:
            self.success_counter += 1
            if self.success_counter >= self.success_thresh:
                self.success_counter = 0 
                self.current_threshold = abs(self.min_en - self.lowest_energy)
  
        return self.current_threshold
    
    def get_current_threshold(self):
        return self.current_threshold

    def update_threshold(self, **kw):
        energy_done = kw.get('energy_done')
        if energy_done:
            self.greedy_shift()


class VanillaCurriculum:
    def __init__(self, config, **kw):
        
        
        
        self.thresholds = config['thresholds']
        self.episodes = config['switch_episodes']
        self.episodes_completed = 0
        self.min_en = kw.get('target_energy') 
        self.current_threshold = config['accept_err']
        self.lowest_energy = self.min_en + self.current_threshold


    def get_current_threshold(self):
        episodes_not_passed = [i for i in range(len(self.episodes)) if self.episodes[i] > self.episodes_completed]
        return self.thresholds[min(episodes_not_passed)]

    def update_threshold(self, **kw):
        self.episodes_completed += 1