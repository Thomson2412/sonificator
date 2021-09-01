(
~instuments = Array.new;
~folder = PathName.new("/mnt/datadrive/projects/thesis/sonificator/data/instuments/");
~folder.filesDo({
	arg file;
	if(file.extension == "wav"){
		file.fullPath.postln;
		~instuments = ~instuments.add(Buffer.read(s, file.fullPath));
	};
});
)

(
SynthDef(\instruments, {
	arg out, freq=60.midicps, amp=1, buf, sustainTime, atk=0, rel=0, gate=1, t_trig=1, start=0, loop=1;
	var sig = PlayBuf.ar(2, buf, (freq / 60.midicps) * BufRateScale.kr(buf), t_trig, start, loop);
	var gateEnv = EnvGen.kr(Env([1, 1, 0], [sustainTime-rel, 0]));
	var env = EnvGen.kr(Env.asr(atk, 1, rel), gate * gateEnv, doneAction: Done.freeSelf);
	sig[0] = DelayN.ar(sig[0], 0.024, 0.024);
	Out.ar(out, sig * env * amp);
}).add;
)

(
Pbind(
	\instrument, \instruments,
	\degree, Pseq([0,3,7], [2,5,8], [3,7,10], [5,8,12], [7,10,14], [8,12,15], [10,14,17]),
	\root, 5,
	\octave, 3,
	\dur, 4,
	\amp, 1,
	\atk, 1,
	\rel, 1,
	\sustainTime, 4,
	\buf, Prand(~instuments, inf),
	\numBufs, b.size,
	\loop, 0,
	\bufPos, 0,
).play;
)

(
Pbind(
	\instrument, \instruments,
	\degree, Pseq([0,3,7,2,5,8,3,7,10,5,8,12,7,10,14,8,12,15,10,14,17], inf),
	\root, 0,
	\octave, 3,
	\dur, 0.2,
	\amp, 1,
	\atk, 0.01,
	\rel, 0.01,
	\sustainTime, 1,
	\buf, Prand(~instuments, inf),
	\numBufs, b.size,
	\loop, 0,
	\bufPos, 0,
).play;
)