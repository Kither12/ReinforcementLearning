import numpy as np;

class Easy21Env():
	def __init__(self) : 
		self.terminate = False
		self.min_range = 1
		self.max_range = 10
		self.red_possibility = 1 / 3
		self.black_possibility = 1 - self.red_possibility

		self.player = Agent()
		self.dealer = Agent()
		self.player.points = abs(self.draw())
		self.dealer.points = abs(self.draw())

	def draw(self) :
		card_num = np.random.randint(self.min_range, self.max_range + 1)
		color = "red" if np.random.random() < self.red_possibility else "black"
		if color == "red" :
			return -card_num
		else :
			return card_num

	def step(self, action) :
		if action == 0 :
			card_num = self.draw()
			self.player.points += card_num

			if self.player.points < 1 or self.player.points > 21 :
				self.terminate = True
				return -1
			else :
				self.terminate = False
				return 0

		else:
			self.terminate = True
			while self.dealer.points < 17 :
				card_num = self.draw()
				self.dealer.points += card_num
				if self.dealer.points < 1 or self.dealer.points > 21 :
					return 1
			if self.player.points > self.dealer.points :
				return 1
			elif self.player.points == self.dealer.points :
				return 0
			else :
				return -1 

	def reset(self) :
		self.player.points = abs(self.draw())
		self.dealer.points = abs(self.draw())


		
	
	

class Agent:
	def __init__(self):
		self.points = 0;


