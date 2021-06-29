(
~input = "/Users/thomas/Documents/sonificator/data/test/Edvard_Munch.txt";
~file = FileReader.read(~input, true)[2].asInteger;
~file.postln;
~file = ~file.normalize(-1, 1);
// ~file.plot;
~wave = Signal.fill(256, {arg i; ~file[i]}).asWavetable;
// w.plot;
~sine = Signal.sineFill(1024, [1], [0]).asWavetable;
// ~sine.plot;
//b.free;
b = Buffer.alloc(s, 512);
b.loadCollection(~wave);
//b.plot;
)

(
SynthDef.new(\vosc, {
	arg buf=0, numBufs=1, bufPos=0,
	freq=440, att=0, dec=1, sus = 0, rel=0.2,
	amp=0.2, revb = 0.33, gate = 1, pan=0;
	var sig, detuneSig, env;
	bufPos = buf + bufPos.min(numBufs - 1);
	env = Env.adsr(att, dec, sus, rel);
	detuneSig = LFNoise1.kr(0.2!8).bipolar(0.2).midiratio;
	sig = VOsc.ar(bufPos, freq * detuneSig);
	sig = Splay.ar(sig, center:pan);
	sig = LeakDC.ar(sig);
	sig = sig * EnvGen.kr(env, gate, doneAction: Done.freeSelf);
	// sig = FreeVerb2.ar(sig, sig, mix: revb, room: 0.5, damp: 0.5, mul: 1.0, add: 0.0);
	sig = LPF.ar(sig, freq: 16000, mul: 1.0, add: 0.0);
	// sig = CrossoverDistortion.ar(sig, amp: bufPos, smooth: 0.5, mul: 1.0, add: 0);
	sig = SmoothDecimator.ar(sig, rate: (44100.0 / 2), smoothing: 0.5, mul: 1.0, add: 0);
	Out.ar(0, sig  * amp);
}).add;
)

(
Pbind(
	\instrument, \vosc,
	\degree, Pseq([[0,2,4,6], [2,6,9,11]]),
	\dur, 3,
	\amp, 0.5,
	\att, 0.5,
	\dec, 3,
	\rel, 1,
	\revb, 0.5,
	\buf, b.bufnum,
	\numBufs, b.size,
	\bufPos, 1,
).play;
)

{CrossoverDistortion.ar(SinOsc.ar([400, 404], 0, 0.2), MouseX.kr(0, 1), MouseY.kr(0, 1))}.play
{SmoothDecimator.ar(SinOsc.ar([400, 404], 0, 0.2), MouseX.kr(0, 44100), MouseY.kr(0, 1))}.play
{Decimator.ar(SinOsc.ar([400, 404], 0, 0.2), MouseX.kr(0, 44100), MouseY.kr(0, 31))}.play