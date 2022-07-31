
import numpy as np


aZcal = 0.03200815494
aYcal = 0.03119266055
aXcal = 0.03165137615

bZcal = 0.03251783894
bYcal = 0.03170234455
bXcal = 0.03267074414

cZcal = 0.03109072375
cYcal = 0.03190621814
cXcal = 0.03251783894

dZcal = 0.03282364934
dYcal = 0.03149847095
dXcal = 0.03272171254

eZcal = 0.03200815494
eYcal = 0.03333333333
eXcal = 0.03236493374

fZcal = 0.03007135576
fYcal = 0.03200815494
fXcal = 0.03231396534

gZcal = 0.03282364934
gYcal = 0.03119266055
gXcal = 0.03282364934

zCal = np.array([aZcal, bZcal, cZcal, dZcal, eZcal, fZcal, gZcal])
yCal = np.array([aYcal, bYcal, cYcal, dYcal, eYcal, fYcal, gYcal])
xCal = np.array([aXcal, bXcal, cXcal, dXcal, eXcal, fXcal, gXcal])


# rest positions with x in radial direction, y in up direction
# in Volts
y_rest_Cal = np.array([1.871, 1.913, 1.908, 1.900, 1.895, 1.903, 1.897])
x_rest_Cal = np.array([1.591, 1.603, 1.588, 1.598, 1.620, 1.603, 1.611])
