import unittest
import torch
from jump_sequence import JumpSequence
class SequenceTests(unittest.TestCase):
 def setUp(self):self.s=JumpSequence(1,'cpu')
 def step(self,x=1.,support=True,free=False,upright=True,reset=False):
  feet=torch.tensor([[[x,0.,0.]]*4]);forces=torch.zeros((1,4,3));forces[:,:,2]=10. if support else 0.
  return self.s.update(feet,forces,torch.tensor([0. if free else 40.]),torch.tensor([x]),torch.tensor([-1. if upright else 0.]),torch.tensor([1.]),torch.tensor([reset]))[0].item()
 def flight(self):
  self.step();self.step(free=True,support=False);return self.step(free=True,support=False)
 def test_valid_sequence_pays_once_after_stability(self):
  self.assertEqual(self.flight(),12.)
  for _ in range(9):self.assertEqual(self.step(x=2.6),0.)
  self.assertEqual(self.step(x=2.6),100.)
  self.assertEqual(self.s.takeoff_count.item(),1)
  self.assertEqual(self.s.landing_count.item(),1)
  self.assertEqual(self.step(x=2.6),0.)
 def test_far_spawn_is_not_jump(self):
  for _ in range(20):self.assertEqual(self.step(x=2.6),0.)
 def test_walking_across_without_flight_is_rejected(self):
  self.step();self.step(x=2.6)
  self.assertEqual(self.s.phase.item(),4)
  for _ in range(20):self.assertEqual(self.step(x=2.6),0.)
 def test_one_frame_air_is_not_transfer(self):
  self.step();self.step(free=True,support=False);self.step(x=2.6)
  self.assertEqual(self.s.phase.item(),4)
 def test_fall_does_not_pay_landing(self):
  self.flight();self.step(x=2.6,upright=False)
  for _ in range(20):self.assertEqual(self.step(x=2.6),0.)
 def test_reset_erases_history(self):
  self.flight();self.step(x=2.6,reset=True)
  for _ in range(20):self.assertEqual(self.step(x=2.6),0.)
if __name__=='__main__':unittest.main()
