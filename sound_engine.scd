s.boot;

(
s.doWhenBooted({
	var buf;
	var wt = Array.newClear(4);
	var sine = Signal.sineFill(1024, [1, 0]).asWavetable;
	var tri = Env([0, 1 , -1, 0], [1, 2, 1], \lin).asSignal(1024).asWavetable;
	var saw = Env([0, 1 , -1, 0], [1, 0, 1], \lin).asSignal(1024).asWavetable;
	var square = Signal.rectWindow(1024, 512).asWavetable;
	/*var plots = Array.newClear(4);
	plots[0] = sine.plot("~sine");
	plots[1] = tri.plot("~tri");
	plots[2] = saw.plot("~saw");
	plots[3] = square.plot("~square");
	plots.do{
		arg plt;
		plt.setProperties(
			// \fontColor, Color.red,
			\plotColor, Color.red,
			// \backgroundColor, Color.black,
			// \gridColorX, Color.white,
		);
		plt.refresh;
	};*/

	wt[0] = sine;
	wt[1] = tri;
	wt[2] = saw;
	wt[3] = square;

	buf = Buffer.allocConsecutive(4, s, 2048);
	buf.do({arg buf, i; buf.loadCollection(wt[i])});

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
		sig = FreeVerb2.ar(sig, sig, mix: revb, room: 0.5, damp: 0.5, mul: 1.0, add: 0.0);
		sig = LPF.ar(sig, freq: 16000, mul: 1.0, add: 0.0);
		Out.ar(0, sig  * amp);
	}).add;

	o.free;
	o = OSCFunc(
		{
			arg msg, time, addr, recvPort;
			var root;
			var octave;
			var scale;
			var amp;
			var duration;
			var release;
			var strum;
			var scale_notes = [[0,2,4,6]];
			var melody_notes = Array.newClear(12); //[0,2,4,6, 0,2,4,5, 0,1,3,5, 0,1,3,4];
			// var melody_notes = [0,2,4,6, 0,2,4,5, 0,1,3,5, 0,1,3,4];
			var edginess;
			var line;

			msg.postln;
			root = msg[1];
			octave = msg[3];
			scale = Scale.major;
			if(msg[3] < 4,{
				scale = Scale.minor;
			},);
			amp = (msg[2] / 1000) - 0.2;
			amp = max(min(amp, 0.16), 0.01);
			duration = msg[4] / 1000;
			release = duration * 0.5;
			strum = 0.08.rand + 0.02;
			edginess = msg[5] * buf.size * 2;
			line = msg[6].normalize(0, 11).collect{arg i; i.nearestInScale(scale)};
			melody_notes.size.do{arg i; melody_notes[i] = line[(i * (line.size / melody_notes.size)).round]};

			//Bass
			Pbind(
				\instrument, \vosc,
				\degree, Pseq([0]),
				\scale, scale,
				\root, root,
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
				\bufPos, edginess,
				\pan, 0,
			).play;

			//Chords
			Pbind(
				\instrument, \vosc,
				\degree, Pseq(scale_notes),
				\scale, scale,
				\root, root,
				\octave, octave,
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
				\degree, Pseq(melody_notes),
				\scale, scale,
				\root, root,
				\octave, octave + 1,
				\dur, Pseq(
					{exprand((duration / melody_notes.size) - 0.1, duration / melody_notes.size)}!melody_notes.size
				),
				\strum, strum,
				\amp, amp,
				\att, 0.08,
				\dec, 0.5,
				\sus, 0,
				\rel, 0.5,
				\buf, buf[0].bufnum,
				\numBufs, buf.size,
				\bufPos, 0,
				\revb, 0.5,
				\pan, rrand(-0.5, 0.5),
			).play;

			Post << "Scale: " << scale.name << Char.nl;
			Post << "Root: " << root << Char.nl;
			Post << "Octave: " << octave << Char.nl;
			Post << "Amp: " << amp << Char.nl;
			Post << "Duration: " << duration << Char.nl;
			Post << "Release: " << release << Char.nl;
			Post << "Strum: " << strum << Char.nl;
			Post << "Edginess: " << edginess << Char.nl;
			Post << "Line: " << line << Char.nl;
			Post << "Scale notes: " << scale_notes << Char.nl;
			Post << "Melody notes: " << melody_notes << Char.nl;
			Post << Char.nl;
		},
		'/low_level_data', nil, 8484);

});
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

[1, 2, 3].normalize(-20, 10);