"""Test custom training geometry without launching Isaac Lab or allocating a GPU."""
import ast,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import trimesh

class TrainingGeometryTests(unittest.TestCase):
    def geometry(self,kind,difficulty,stage="first"):
        tree=ast.parse(Path('terrain_train.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='obstacle')
        scope={'np':np,'trimesh':trimesh,'a':SimpleNamespace(kind=kind,stage=stage),'terrain_records':[]}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'terrain_train.py','exec'),scope)
        meshes,origin=scope['obstacle'](difficulty,None)
        return meshes,origin
    def test_gap_has_no_surface_at_bank_height(self):
        for d in [0.,.5,1.]:
            meshes,origin=self.geometry('gap',d)
            width=.2+.25*d; midpoint=3.5+width/2
            self.assertEqual(origin.tolist(),[2.,4.,0.])
            for mesh in meshes:
                low,high=mesh.bounds
                self.assertFalse(low[0]<midpoint<high[0] and high[2]>-.39)
            self.assertTrue(any(abs(m.bounds[1,2]+.4)<1e-8 for m in meshes))
    def test_jump_stage_gap_geometry(self):
        for d in [0.,.5,1.]:
            meshes,origin=self.geometry('gap',d,'jump')
            width=.4+.3*d
            self.assertAlmostEqual(meshes[1].bounds[0,0]-3.5,width)
            midpoint=3.5+width/2
            self.assertFalse(any(m.bounds[0,0]<midpoint<m.bounds[1,0] and m.bounds[1,2]>-.39 for m in meshes))

    def test_hard_stage_gap_geometry(self):
        for d in [0.,.5,1.]:
            meshes,_=self.geometry("gap",d,"hard")
            width=.55+.35*d
            self.assertAlmostEqual(meshes[1].bounds[0,0]-3.5,width)
            midpoint=3.5+width/2
            self.assertFalse(any(m.bounds[0,0]<midpoint<m.bounds[1,0] and m.bounds[1,2]>-.39 for m in meshes))

    def test_step_riser_and_width(self):
        for d in [0.,.5,1.]:
            meshes,origin=self.geometry('step',d)
            low,high=meshes[1].bounds
            self.assertAlmostEqual(low[0]-origin[0],1.5)
            self.assertAlmostEqual(high[2],.18+.12*d)
            self.assertAlmostEqual(high[1]-low[1],1.8)

if __name__=='__main__':unittest.main()
