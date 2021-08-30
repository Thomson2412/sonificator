(
~instuments = Array.new;
~folder = PathName.new("/Users/thomas/Documents/sonificator/data/instuments/");
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
	\degree, Pseq([0,4,7,11], [2,5,9,12], [4,7,11,14], [5,9,12,16], [7,11,14,17], [9,12,16,19], [11,14,17,21]),
	\root, 0,
	\octave, 4,
	\dur, 4,
	\amp, 4,
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
	\degree, Pseq([0,4,7,11], inf),
	\root, 0,
	\octave, 3,
	\dur, 0.5,
	\amp, 10,
	\atk, 0.1,
	\rel, 0.1,
	\sustainTime, 1,
	\buf, Prand(~instuments, inf),
	\numBufs, b.size,
	\loop, 0,
	\bufPos, 0,
).play;
)