s.boot;

(
s.doWhenBooted({
	var buf;
	var wt = Array.newClear(4);
	var sine = Signal.sineFill(1024, [1, 0]).asWavetable;
	var tri = Env([0, 1 , -1, 0], [1, 2, 1], \lin).asSignal(1024).asWavetable;
	var saw = Env([0, 1 , -1, 0], [1, 0, 1], \lin).asSignal(1024).asWavetable;
	var square = Signal.rectWindow(1024, 512).asWavetable;

	wt[0] = sine;
	wt[1] = tri;
	wt[2] = saw;
	wt[3] = square;

	buf = Buffer.allocConsecutive(4, s, 2048);
	buf.do({arg buf, i; buf.loadCollection(wt[i])});

	SynthDef.new(\vosc, {
		arg buf=0, numBufs=1, bufPos=0, freq=440, att=0, dec=1, sus = 0, rel=0.2, amp=0.2, gate = 1, pan=0;
		var sig, detuneSig, env;
		bufPos = buf + bufPos.min(numBufs - 1);
		env = Env.adsr(att, dec, sus, rel);
		detuneSig = LFNoise1.kr(0.2!8).bipolar(0.2).midiratio;
		sig = VOsc.ar(bufPos, freq * detuneSig);
		sig = Splay.ar(sig, center:pan);
		sig = LeakDC.ar(sig);
		sig = sig * EnvGen.kr(env, gate, doneAction: Done.freeSelf);
		sig = FreeVerb2.ar(sig, sig, mix: 0.33, room: 0.5, damp: 0.5, mul: 1.0, add: 0.0);
		sig = LPF.ar(sig, freq: 16000, mul: 1.0, add: 0.0);
		Out.ar(0, sig  * amp);
	}).add;

	o.free;
	o = OSCFunc(
		{
			arg msg, time, addr, recvPort;
			var scale;
			var amp;
			var duration;
			var release;
			var strum;
			var scale_notes = [[0,2,4,6]];
			var melody_notes = [0,2,4,6, 0,2,4,5, 0,1,3,5, 0,1,3,4];
			var edginess;
			var line;
			msg.postln;
			scale = Scale.major;
			if(msg[3] < 4,{
				scale = Scale.minor;
			},);
			amp = (msg[2] / 1000) - 0.1;
			amp = max(min(amp, 0.2), 0.01);
			duration = msg[4] / 1000;
			release = duration * 0.5;
			strum = 0.08.rand + 0.02;
			edginess = msg[5] * ~buf.size * 2;
			line = (msg[6] - 50) / 10;
			//Bass
			Pbind(
				\instrument, \vosc,
				\degree, Pseq([0]),
				\scale, scale,
				\root, msg[1],
				\octave, 3,
				\dur, duration + 1,
				\strum, 0,
				\amp, amp,
				\att, duration * 0.5,
				\dec, duration,
				\sus, 0,
				\rel, release,
				\buf, buf[0].bufnum,
				\numBufs, buf.size,
				\bufPos, 1.5,
				\pan, 0,
			).play;
			//Chords
			Pbind(
				\instrument, \vosc,
				\degree, Pseq(scale_notes),
				\scale, scale,
				\root, msg[1],
				\octave, msg[3],
				\dur, duration + 1,
				\strum, strum,
				\amp, amp,
				\att, duration * 0.5,
				\dec, duration,
				\sus, 0,
				\rel, release,
				\buf, buf[0].bufnum,
				\numBufs, buf.size,
				\bufPos, edginess,
				\pan, rrand(-0.5, 0.5),
			).play;
			//Melody
			Pbind(
				\instrument, \vosc,
				\degree, Pseq(melody_notes.scramble),
				\scale, scale,
				\root, msg[1],
				\octave, msg[3] + 1,
				\dur, Pseq({min(0.2, max(0.12, rrand(0.15, 0.2) - (edginess / 100)))}!melody_notes.size),
				\strum, strum,
				\amp, amp,
				\att, 0.1,
				\dec, 0.5,
				\sus, 0,
				\rel, 0.5,
				\buf, buf[0].bufnum,
				\numBufs, buf.size,
				\bufPos, 0,
				\pan, rrand(-0.5, 0.5),
			).play;
			Post << "Scale: " << scale.name << Char.nl;
			Post << "Root: " << msg[1] << Char.nl;
			Post << "Octave: " << msg[3] << Char.nl;
			Post << "Amp: " << amp << Char.nl;
			Post << "Duration: " << duration << Char.nl;
			Post << "Release: " << release << Char.nl;
			Post << "Strum: " << strum << Char.nl;
			Post << "Scale notes: " << scale_notes << Char.nl;
			Post << "Edginess: " << edginess << Char.nl;
			Post << "Line: " << line << Char.nl;
			Post << Char.nl;
		},
		'/low_level_data', nil, 8484);

});
)



(
~sine.plot("~sine");
~tri.plot("~tri");
~saw.plot("~saw");
~square.plot("~square");
)





{min(0.2, max(0.15, rrand(0.15, 0.2) - ))}!12
(
var line;
var plot;
o.free;
o = OSCFunc(
	{
		arg msg, time, addr, recvPort;
		msg.postln;
		line = msg[5];
		line = (line - 50) / 100;
		line.postln;
		l = line
	},
	'/test', nil, 8484);
)
m = [0,2,4,6, 0,2,4,5, 0,1,3,5, 0,1,3,4];
(
Pbind(
	\instrument, \vosc,
	\degree, Pseq(m),
	\scale, Scale.major,
	\root, 0,
	\octave, 6,
	\dur, Pseq([rrand(0.2, 0.5)], m.size),
	// \strum, 0.08,
	\amp, 0.2,
	\att, 0.1,
	\dec, 0.5,
	\sus, 0,
	\rel, 0.5,
	\buf, ~buf[0].bufnum,
	\numBufs, ~buf.size,
	\bufPos, 0,
	\pan, rrand(-0.5, 0.5),
).play;
)

a = Scale.major;
a.name;
a.degrees;
a.semitones;
a.ratios;

b = Scale.minor;
b.name;
b.degrees;
b.semitones;
b.ratios;

Env.adsr(0.1, 0.1, 0, 0.2).plot;
s.nodeMap

