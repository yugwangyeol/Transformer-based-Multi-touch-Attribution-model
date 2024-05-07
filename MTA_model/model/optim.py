import numpy as np

class ScheduledAdam():
    def __init__(self,optimizer, hidden_dim, warm_steps):
        self.init_lr = np.power(hidden_dim,-0.5) #어진 배열의 각 요소를 지수승으로 계산
        self.optimizer = optimizer
        self.current_steps = 0
        self.warm_steps = warm_steps

    def step(self):
        # 현재 걸음 수 정보를 사용하여 학습률 업데이트
        self.current_steps += 1
        lr = self.init_lr*self.get_scale()

        for p in self.optimizer.param_groups:
            p['lr'] = lr
        
        self.optimizer.step()
    
    def zero_grade(self):
        self.optimizer.zero_grade()

    def get_scale(self):
        return np.min([
            np.power(self.current_steps,-0.5), # warm step이 끝난후
            self.current_steps*np.power(self.warm_steps,-0.5) # warm step동안은 뒷부분 사용
        ])