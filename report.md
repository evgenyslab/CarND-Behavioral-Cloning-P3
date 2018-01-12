
used nvidia framework with image cropping & 3 image sets.
train with vanilla, seemed to work - may be over fitting
gathered the following data for retraining:
2 loops forward driven well + 1 loop with swerving recovery
same as above but in reverse.

Trained model with vanila data works well on the track, most likely overfits.
Trained model with new data results in more car sway which would be due to the reverse runs.
low resolution of images suggest the sway occurs due to networks attempt to locate the shoulder of the road, while vanila model just encodes the whole track.


