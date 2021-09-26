import torch
import torch.nn as nn
import torch.nn.functional as F

class KLLoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.KLLoss = nn.KLDivLoss()

	def forward(self, output, target):
		'''
		Output: (N,*) \n
		Target: (N,*) \n
		'''
		output = torch.log(output)  # Invert softmax
		# target = torch.log(target) # Invert softmax
		# How output distribution differs from target distribution
		return self.KLLoss(output, target)


class CELoss(nn.Module):
	def __init__(self, ignore_index=-1):
		super().__init__()
		self.CELoss = nn.CrossEntropyLoss(ignore_index=ignore_index)

	def forward(self, output, target):
		'''
		Output: (N,*,C) \n
		Target: (N,*) \n
		'''
		output = torch.log(output)  # Invert softmax
		output = output.reshape(-1, output.shape[-1])  # (*,C)
		target = target.reshape(-1).long()  # (*)
		return self.CELoss(output, target)


class CELossSame(nn.Module):
	def __init__(self, ignore_index=-1):
		super().__init__()
		self.CELoss = nn.CrossEntropyLoss(ignore_index=ignore_index)

	def forward(self, outputs, target):
		'''
		Output: (N,*,C) \n
		Target: (N,*) \n
		'''
		output_img = torch.log(outputs[0]) # Invert softmax
		output_txt = torch.log(outputs[1])
		output_sen = torch.log(outputs[2])

		output_img = output_img.reshape(-1, output_img.shape[-1]) # (*,C)
		output_txt = output_txt.reshape(-1, output_txt.shape[-1]) # (*,C)
		output_sen = output_sen.reshape(-1, output_sen.shape[-1]) # (*,C)
		target = target.reshape(-1).long() # (*)
		return self.CELoss(output_img, target) + self.CELoss(output_txt, target) + self.CELoss(output_sen, target)

class CELossShift(nn.Module):
	def __init__(self, ignore_index=-1):
		super().__init__()
		self.CELoss = CELoss(ignore_index=ignore_index)

	def forward(self, output, target):
		'''
		Output: (N,*,C) \n
		Target: (N,*) \n
		'''
		output = output[:,:-1,:] # (* - 1,C)
		target = target[:,1:] # (* - 1)
		return self.CELoss(output, target)

class CELossTotal(nn.Module):
	def __init__(self, ignore_index=-1):
		super().__init__()
		self.CELoss = CELoss()
		self.CELossShift = CELossShift(ignore_index=ignore_index)

	def forward(self, output, target):
		return self.CELossShift(output[0], target[0]) + self.CELoss(output[1], target[1])

class CELossTotalEval(nn.Module):
	def __init__(self, ignore_index=-1):
		super().__init__()
		self.CELoss = CELoss()
		self.CELossShift = CELossShift(ignore_index=ignore_index)

	def forward(self, output, target):
		return self.CELossShift(output[0], target[0]) + self.CELoss(output[1], target[1]) + self.CELoss(output[2], target[1])

class CELossTransfer(nn.Module):
	def __init__(self, ignore_index=-1):
		super().__init__()
		self.CELoss = CELoss()
		self.CELossShift = CELossShift(ignore_index=ignore_index)

	def forward(self, output, target):
		return self.CELossShift(output[0], target[0]) # + self.CELoss(output[1], target[1])