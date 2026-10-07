import math
import pytest
from games.fancraft.journey import Journey

@pytest.mark.parametrize('angle',[0,math.pi/2,math.pi,3*math.pi/2])
@pytest.mark.parametrize('facing',[0,.7,math.pi])
def test_line_dodge_moves_out_the_closest_side_for_every_facing(angle,facing):
    j=object.__new__(Journey);fx,fz=-math.sin(angle),-math.cos(angle)
    marker={'markerType':'line_aoe','position':{'x':0,'z':0},'yaw':angle,'length':20,'width':4,'radius':20,'remainingMs':2000}
    for side in [-1,1]:
        x,z=fx*10+fz*side,fz*10-fx*side
        forward,strafe=j._dodge_vector({'self':{'x':x,'z':z,'yaw':facing},'markers':[marker]})
        wx,wz=-math.sin(facing)*forward+math.cos(facing)*strafe,-math.cos(facing)*forward-math.sin(facing)*strafe
        assert abs(wx*fx+wz*fz)<1e-8
        assert (wx*fz-wz*fx)*side>.99

@pytest.mark.parametrize('x,z',[(4,-10),(0,2),(0,-22)])
def test_line_safe_areas_do_not_fall_through_to_circle_dodging(x,z):
    j=object.__new__(Journey)
    m={'markerType':'line_aoe','position':{'x':0,'z':0},'yaw':0,'length':20,'width':4,'radius':20,'remainingMs':2000}
    assert j._dodge_vector({'self':{'x':x,'z':z,'yaw':0},'markers':[m]}) is None
