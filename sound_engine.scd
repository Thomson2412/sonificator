/*
root = [0]
scale = [1]
steps = [2]
hue = [3]
saturation = [4]
intensity = [5]
duration = [6]
edginess = [7]
line = [8]
*/

(
fork{

	if (thisProcess.argv[0] != nil) {
		i = thisProcess.argv[0];
	} {
		i = "/mnt/datadrive/projects/thesis/sonificator/data/test/painting10.txt"
	};

	if (thisProcess.argv[1] != nil) {
		o = thisProcess.argv[1];
	};

	i.postln;
	o.postln;

	s.waitForBoot({
		var buf;
		var wt = Array.newClear(4);
		var sine = Signal.sineFill(1024, [1, 0]).asWavetable;
		var tri = Env([0, 1 , -1, 0], [1, 2, 1], \lin).asSignal(1024).asWavetable;
		var saw = Env([0, 1 , -1, 0], [1, 0, 1], \lin).asSignal(1024).asWavetable;
		var square = Signal.rectWindow(1024, 512).asWavetable;

		var maj = [[0,4,7,11], [2,5,9,12], [4,7,11,14], [5,9,12,16], [7,11,14,17], [9,12,16,19], [11,14,17,21]];
		var min = [[0,3,7,10], [2,5,8,12], [3,7,10,14], [5,8,12,15], [7,10,14,17], [8,12,15,19], [10,14,17,20]];

		var file = FileReader.read(i, true);

		var root;
		var scale;

		var steps;

		var octaves;


		var scaleSteps;
		var chords;

		var lines;
		var melody_notes;
		var melody_notes_amount;
		var melody_notes_durations;
		var melody_note_count;
		var melody_octaves;
		var melody_amps;
		var melody_pans;

		var edginess;

		var amps;
		var durations;
		var releases;
		var strums;
		var pans;

		wt[0] = sine;
		wt[1] = tri;
		wt[2] = square;
		wt[3] = saw;

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

		steps = file[2][0].asInteger;

		root = file[0][0].asInteger;

		scaleSteps = file[3].asInteger;

		if(file[1][0].asInteger == 0) {
			scale = Scale.minor;
			chords = scaleSteps.collect({arg i; min[i]});
		} {
			scale = Scale.major;
			chords = scaleSteps.collect({arg i; maj[i]});
		};

		octaves = file[5].asInteger;

		edginess = file[7].asFloat.collect({arg i; min(i * buf.size * 2, buf.size - 1.1)});

		amps = file[4].asInteger / 1000;
		amps = max(min(amps, 0.1), 0.001);
		durations = file[6].asInteger;
		releases = durations * 0.5;
		strums = Array.fill(steps, {0.08.rand + 0.02});
		pans = Array.fill(steps, {rrand(-0.5, 0.5)});

		lines = file[8..23].asInteger;
		melody_notes_amount = 12;
		melody_notes = Array.newClear(steps * melody_notes_amount);
		melody_notes_durations = Array.newClear(steps * melody_notes_amount);
		melody_octaves = Array.newClear(steps * melody_notes_amount);
		melody_amps = Array.newClear(steps * melody_notes_amount);
		melody_pans = Array.newClear(steps * melody_notes_amount);

		melody_note_count = 0;
		steps.do({
			arg s;
			// s.postln;
			melody_notes_amount.do({
				arg i;
				// i.postln;
				melody_notes[melody_note_count] = lines[s][(i * (lines[s].size / melody_notes_amount)).trunc];
				melody_notes_durations[melody_note_count] = durations[s] / melody_notes_amount;
				// melody_notes_durations[melody_note_count] = exprand((durations[s] / melody_notes_amount) - 0.15, (durations[s] / melody_notes_amount) + 0.15);
				melody_octaves[melody_note_count] = octaves[s] + 1;
				melody_amps[melody_note_count] = amps[s];
				melody_pans[melody_note_count] = rrand(-0.5, 0.5);

				melody_note_count = melody_note_count + 1;
			});
		});

		if(o != nil){
			s.record(duration: durations.sum + 1, path: o);
		};

		//Bass
		Pbind(
			\instrument, \vosc,
			\degree, Pseq(chords.collect({arg i; i[0]})),
			\root, root,
			\octave, 2,
			\dur, Pseq(durations),
			\strum, 0,
			\amp, Pseq(amps),
			\att, Pseq(durations * 0.5),
			\dec, Pseq(durations),
			\sus, 0,
			\rel, Pseq(releases),
			\buf, buf[0].bufnum,
			\numBufs, buf.size,
			\bufPos, Pseq(edginess),
			\pan, 0,
		).play;

		//Chords
		Pbind(
			\instrument, \vosc,
			\degree, Pseq(chords),
			\root, root,
			\octave, Pseq(octaves),
			\dur, Pseq(durations),
			\strum, Pseq(strums),
			\amp, Pseq(amps),
			\att, Pseq(durations * 0.5),
			// \dec, Pseq(durations),
			\sus, 0,
			\rel, Pseq(releases),
			\buf, buf[0].bufnum,
			\numBufs, buf.size,
			\bufPos, Pseq(edginess),
			\revb, 0.6,
			\pan, Pseq(pans),
		).play;

		//Melody
		Pbind(
			\instrument, \vosc,
			\degree, Pseq(melody_notes),
			\scale, scale,
			\root, root,
			\octave, Pseq(melody_octaves),
			\dur, Pseq(melody_notes_durations),
			\strum, 0,
			\amp, Pseq(melody_amps),
			\att, 0.08,
			\dec, 0.5,
			\sus, 0,
			\rel, 0.5,
			\buf, buf[0].bufnum,
			\numBufs, buf.size,
			\bufPos, 0,
			\revb, 0.5,
			\pan, Pseq(melody_pans),
		).play;


		Post << "Scale: " << scale.name << Char.nl;
		Post << "Root: " << root << Char.nl;
		Post << "Steps: " << root << Char.nl;
		Post << "Octaves: " << octaves << Char.nl;
		Post << "ScaleSteps: " << scaleSteps << Char.nl;
		Post << "Chords: " << chords << Char.nl;

		Post << "Lines: " << lines << Char.nl;
		Post << "Melody notes: " << melody_notes << Char.nl;
		Post << "Melody notes durations: " << melody_notes_durations << Char.nl;
		Post << "Melody octaves: " << melody_octaves << Char.nl;
		Post << "Melody amps: " << melody_amps << Char.nl;
		Post << "Melody pans: " << melody_pans << Char.nl;

		Post << "Edginess: " << edginess << Char.nl;

		Post << "Amps: " << amps << Char.nl;
		Post << "Durations: " << durations << Char.nl;
		Post << "Releases: " << releases << Char.nl;
		Post << "Strums: " << strums << Char.nl;
		Post << "Pans: " << strums << Char.nl;
		Post << Char.nl;

		if(o != nil){
			(durations.sum + 5).wait;
			0.exit;
		};
	});
})