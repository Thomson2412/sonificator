(
~input = "/mnt/datadrive/projects/thesis/sonificator/data/test/allegorical_painting-Arthur_Streeton-Spirit_of_the_drought-1895-65154_hist.txt";
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

// Reading the file
f = SoundFile.openRead("/mnt/datadrive/projects/thesis/Datasets/Audio/FSD50K/FSD50K.dev_audio/158986.wav");
f.sampleFormat;
f.numChannels;
c = f.cue;
c.play;
c.pause;
c.resume;
c.stop;
f.close;
f.free;

(
SynthDef(\audioFileMono, {
	arg out, amp=1, bufnum, sustainTime, atk=0, rel=0, gate=1, rate=1, t_trig=1, start=0, loop=1;
	var sig = PlayBuf.ar(1, bufnum, BufRateScale.kr(bufnum) * rate, t_trig, start, loop)!2;
	var gateEnv = EnvGen.kr(Env([1, 1, 0], [sustainTime-rel, 0]));
    var env = EnvGen.kr(Env.asr(atk, 1, rel), gate * gateEnv, doneAction: Done.freeSelf);
	Out.ar(out, sig * env * amp);
}).add;
)
k = SoundFile.openRead("/mnt/datadrive/projects/thesis/Datasets/Audio/FSD50K/FSD50K.dev_audio/158986.wav").asBuffer;
k.query;
Synth.new(\audioFileMono, [\bufnum, k.bufnum, \sustainTime, 30]);

