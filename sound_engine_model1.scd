(
fork{
	//File offsets
	~root = 0;
	~scale = 1;
	~melody_note_amount = 2;
	~waveform = 3;
	~scene_audio_path = 4;
	~things = 5;
	~steps = 6;
	~hue = 7;
	~saturation = 8;
	~intensity = 9;
	~pan = 10;
	~duration = 11;
	~edginess = 12;
	~line = 13;

	~vbus = Bus.audio(s, 2);

	if (thisProcess.argv[0] != nil) {
		~input_file = thisProcess.argv[0];
	} {
		~input_file = "/mnt/datadrive/projects/thesis/sonificator/data/test/paintings/animal_painting-Eugene_Boudin-Cows_in_a_Pasture-1883-50445.txt";
		// ~input_file = "/mnt/datadrive/projects/thesis/sonificator/data/test/paintings/allegorical_painting-Arthur_Streeton-Spirit_of_the_drought-1895-65154.txt";
		// ~input_file = "/mnt/datadrive/projects/thesis/sonificator/data/test/paintings/battle_painting-Konstantin_Korovin-Polovtsian_camp_-1914-70568.txt";

		// ~input_file = "/mnt/datadrive/projects/thesis/sonificator/data/test/paintings/Hunters_in_the_Snow_(Winter)_by_Pieter_Bruegel_the_Elder.txt";
		// ~input_file = "/mnt/datadrive/projects/thesis/sonificator/data/test/paintings/Corot_Venise,_La_Piazetta.txt";

		// ~input_file = "/Users/thomas/Documents/sonificator/data/presentation_scene/animal_painting-Eugene_Boudin-Cows_in_a_Pasture-1883-50445.txt";
	};

	if (thisProcess.argv[1] != nil) {
		~output_file = thisProcess.argv[1];
	};

	~input_file.postln;
	~output_file.postln;

	// Server.killAll();
	Buffer.freeAll(s);
	CmdPeriod.run;
	s.waitForBoot({
		var buf;
		var wt = Array.newClear(3);
		var wave;
		var sine = Signal.sineFill(512, [1, 0]).asWavetable;
		var tri = Env([0, 1 , -1, 0], [1, 2, 1], \lin).asSignal(512).asWavetable;
		var saw = Env([0, 1 , -1, 0], [1, 0, 1], \lin).asSignal(512).asWavetable;
		var square = Signal.rectWindow(512, 256).asWavetable;

		var instruments = Array.new;
		var instruments_melody = Array.new;

		// var maj = [[0,4,7,11], [2,5,9,12], [4,7,11,14], [5,9,12,16], [7,11,14,17], [9,12,16,19], [11,14,17,21]];
		// var min = [[0,3,7,10], [2,5,8,12], [3,7,10,14], [5,8,12,15], [7,10,14,17], [8,12,15,19], [10,14,17,20]];

		var maj = [[0,4,7], [2,5,9], [4,7,11], [5,9,12], [7,11,14], [9,12,16], [11,14,17]];
		var min = [[0,3,7], [2,5,8], [3,7,10], [5,8,12], [7,10,14], [8,12,15], [10,14,17]];

		var file = FileReader.read(~input_file, true);

		var root;
		var scale;

		var scene_path;
		var scene_buffer;
		var scene_args;
		var wait_for_scene_load = true;

		var amount_of_things;

		var steps;

		var octaves;


		var scale_steps;
		var chords;

		var lines;
		var melody_notes;
		var melody_notes_amount_local;
		var melody_notes_durations;
		var melody_note_count;
		var lastMelodyNote;
		var melody_octaves;
		var melody_amps;
		var melody_pans;

		var edginess;

		var amps;
		var durations;
		var releases;
		var strums;
		var pans;

		wave = file[~waveform].asInteger.normalize(-1, 1);
		wave = Signal.fill(512, {arg i; wave[i]}).asWavetable;
		wt[0] = sine;
		wt[1] = wave;
		// wt[1] = tri;
		// wt[2] = square;
		wt[2] = saw;

		/*wt.do({
		arg w;
		var plt = w.plot;
		plt.setProperties(
		// \fontColor, Color.red,
		\plotColor, Color.green,
		// \backgroundColor, Color.black,
		// \gridColorX, Color.white,
		);
		plt.refresh;
		});*/
		buf = Buffer.allocConsecutive(3, s, 1024);
		buf.do({arg buf, i; buf.loadCollection(wt[i])});

		SynthDef.new(\vosc, {
			arg buf=0, numBufs=1, bufPos=0,
			freq=440, atk=0, dec=1, sus=0, rel=0.2,
			amp=0.2, gate=1, pan=0,
			out=0, fx=0, fxsend=(-25);
			var sig, detuneSig, env;
			bufPos = buf + bufPos.min(numBufs - 1).max(0);
			env = Env.adsr(atk, dec, sus, rel);
			detuneSig = LFNoise1.kr(0.2!2).bipolar(0.2).midiratio;
			sig = VOsc.ar(bufPos, freq * detuneSig);
			sig = Splay.ar(sig, center:pan);
			sig = LeakDC.ar(sig);
			sig = sig * EnvGen.kr(env, gate, doneAction: Done.freeSelf);
			sig[1] = DelayN.ar(sig[1], 0.016, 0.016);
			sig = LPF.ar(sig, freq: 16000, mul: 1.0, add: 0.0);
			// sig = CrossoverDistortion.ar(sig, amp: bufPos, smooth: 1.0, mul: 1.0, add: 0);
			// sig = SmoothDecimator.ar(in, rate: 44100.0, smoothing: 0.5, mul: 1.0, add: 0);
			// sig = Decimator.ar(sig, rate: 44100.0 * bufPos, bits: 24, mul: 1.0, add: 0);
			Out.ar(0, sig  * amp);
			Out.ar(fx, sig * fxsend.dbamp);
		}).add;

		SynthDef(\audioFileMono, {
			arg out, fx=0, fxsend=(-25), amp=1, bufnum, sustainTime,
			atk=0, dec=1, sus=0, rel=0,
			gate=1, rate=1, t_trig=1, start=0, loop=1;
			var sig = PlayBuf.ar(1, bufnum, BufRateScale.kr(bufnum) * rate, t_trig, start, loop)!2;
			var gateEnv = EnvGen.kr(Env([1, 1, 0], [sustainTime, 0]));
			var env = EnvGen.kr(Env.adsr(atk, dec, sus, rel, 1), gate * gateEnv, doneAction: Done.freeSelf);
			sig = CompanderD.ar(sig, sig, thresh: 0.4, slopeBelow: 0.5, slopeAbove: 0.1, clampTime: 0.01, relaxTime: 0.01);
			sig = FreeVerb2.ar(sig, sig, mix: 0.5, room: 0.2, damp: 0.5);
			sig[0] = DelayN.ar(sig[0], 0.024, 0.024);
			sig = LPF.ar(sig, freq: 4000);
			sig = HPF.ar(sig, freq: 220);
			Out.ar(out, sig * env * amp);
			// Out.ar(fx, sig * env * fxsend.dbamp);
		}).add;

		SynthDef(\audioFileStereo, {
			arg out, fx=0, fxsend=(-25), amp=1, bufnum, sustainTime,
			atk=0, dec=1, sus=0, rel=0,
			gate=1, rate=1, t_trig=1, start=0, loop=1;
			var sig = PlayBuf.ar(2, bufnum, BufRateScale.kr(bufnum) * rate, t_trig, start, loop);
			var gateEnv = EnvGen.kr(Env([1, 1, 0], [sustainTime, 0]));
			var env = EnvGen.kr(Env.adsr(atk, dec, sus, rel, 1), gate * gateEnv, doneAction: Done.freeSelf);
			sig = CompanderD.ar(sig, sig, thresh: 0.4, slopeBelow: 0.5, slopeAbove: 0.1, clampTime: 0.01, relaxTime: 0.01);
			sig = FreeVerb2.ar(sig, sig, mix: 0.5, room: 0.2, damp: 0.5);
			sig[0] = DelayN.ar(sig[0], 0.024, 0.024);
			sig = LPF.ar(sig, freq: 4000);
			sig = HPF.ar(sig, freq: 220);
			Out.ar(out, sig * env * amp);
			// Out.ar(fx, sig * env * fxsend.dbamp);
		}).add;

		SynthDef(\reverb, {
			arg in=0, out=0, dec=4, lpf=1500;
			var sig;
			sig = In.ar(in, 2).sum;
			sig = DelayN.ar(sig, 0.03, 0.03);
			sig = CombN.ar(sig, 0.1, {Rand(0.01,0.099)}!32, dec);
			sig = SplayAz.ar(2, sig);
			sig = LPF.ar(sig, lpf);
			5.do{sig = AllpassN.ar(sig, 0.1, {Rand(0.01,0.099)}!2, 3)};
			sig = LPF.ar(sig, lpf);
			sig = LeakDC.ar(sig);
			Out.ar(out, sig);
		}).add;

		s.sync;

		Synth(\reverb, [\in, ~vbus]);

		s.sync;

		steps = file[~steps][0].asInteger;

		root = file[~root][0].asInteger;

		scene_path = file[~scene_audio_path][0];

		amount_of_things = file[~things][0].asInteger;

		scale_steps = file[~hue].asInteger;

		if(file[~scale][0].asInteger == 0) {
			scale = Scale.melodicMinor;
			chords = scale_steps.collect({arg i; min[i]});
		} {
			scale = Scale.major;
			chords = scale_steps.collect({arg i; maj[i]});
		};

		octaves = file[~intensity].asInteger;

		edginess = file[~edginess].asFloat.collect({arg i; min(i * buf.size * 2, buf.size - 1.1)});
		// edginess = file[9].asFloat.normalize(0, 1.99);
		// edginess = file[9].asFloat.collect({arg i; 0.9});

		amps = file[~saturation].asInteger / 1000;
		amps = max(min(amps, 0.1), 0.001);
		durations = file[~duration].asFloat;
		releases = durations * 0.5;
		strums = Array.fill(steps, {0.08.rand + 0.02});
		pans = file[~pan].asFloat;

		if(scene_path != " " && scene_path != "") {
			scene_buffer = Buffer.read(s, scene_path, action: {wait_for_scene_load = false;});
			scene_buffer = scene_buffer.normalize;
			scene_args = [
				\bufnum, scene_buffer.bufnum,
				\atk, 4,
				\dec, 0,
				\sus, 1,
				\rel, 4,
				\sustainTime, durations.sum,
				\amp, max(amps.minItem() + 0.12, 0.0006)];
		} {
			wait_for_scene_load = false;
		};

		lines = steps.collect({arg s; file[s + ~line].asInteger;});
		melody_notes_amount_local = file[~melody_note_amount][0].asInteger;
		melody_notes = Array.newClear(steps * melody_notes_amount_local);
		melody_notes_durations = Array.newClear(steps * melody_notes_amount_local);
		melody_octaves = Array.newClear(steps * melody_notes_amount_local);
		melody_amps = Array.newClear(steps * melody_notes_amount_local);
		melody_pans = Array.newClear(steps * melody_notes_amount_local);

		melody_note_count = 0;
		steps.do({
			arg s;
			var tempDurArr = Array.newClear(melody_notes_amount_local - 1);
			// s.postln;
			melody_notes_amount_local.do({
				arg i;
				var randDur = exprand((durations[s] / melody_notes_amount_local) - 0.1, (durations[s] / melody_notes_amount_local) + 0.05);
				var lastDur = 0;
				var melodyNote;
				var newMelodyNote;
				// i.postln;

				melody_notes_durations[melody_note_count] = durations[s] / melody_notes_amount_local;
				/*				if(i == (melody_notes_amount_local - 1)) {
				lastDur = durations[s] - tempDurArr.sum();
				if(lastDur < 0){
				melody_notes_durations[melody_note_count - 1] = melody_notes_durations[melody_note_count - 1] - lastDur;
				} {
				melody_notes_durations[melody_note_count] = durations[s] - tempDurArr.sum();
				};
				} {
				melody_notes_durations[melody_note_count] = randDur;
				tempDurArr[i] = randDur;
				};*/

				melodyNote = lines[s][(i * (lines[s].size / melody_notes_amount_local)).trunc];
				if(melodyNote == lastMelodyNote) {newMelodyNote = \rest;} {newMelodyNote = melodyNote};
				melody_notes[melody_note_count] = newMelodyNote;
				lastMelodyNote = melodyNote;
				//.nearestInScale(chords[s]);

				melody_octaves[melody_note_count] = octaves[s] + 1;
				melody_amps[melody_note_count] = amps[s] + 0.08;
				melody_pans[melody_note_count] = pans[s] + rrand(-0.1, 0.1);

				melody_note_count = melody_note_count + 1;
			});
		});

		//Wait to make sure scene_buffer is loaded
		while({wait_for_scene_load}, {0.1.wait; "Waiting on scene buffer".postln});

		if(~output_file != nil){
			s.record(duration: durations.sum + 2, path: ~output_file);
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
			\atk, Pseq(durations * 0.5),
			\dec, Pseq(durations - releases),
			\sus, 0,
			\rel, Pseq(releases),
			\buf, buf[0].bufnum,
			\numBufs, buf.size,
			\bufPos, Pseq(edginess),
			\pan, 0,
			\fx, ~vbus,
			\fxsend, -30,
		).play;

		//Chords
		Pbind(
			\instrument, \vosc,
			\degree, Pseq(chords),
			\root, root,
			\octave, Pseq(octaves),
			\dur, Pseq(durations),
			\strum, 0, // Pseq(strums),
			\amp, Pseq(amps),
			\atk, Pseq(durations * 0.5),
			\dec, Pseq(durations - releases),
			\sus, 0,
			\rel, Pseq(releases),
			\buf, buf[0].bufnum,
			\numBufs, buf.size,
			\bufPos, Pseq(edginess),
			\pan, Pseq(pans),
			\fx, ~vbus,
			\fxsend, -20,
		).play;

		if(melody_notes_amount_local > 0){
			//Melody
			steps.do({arg i; melody_notes_amount_local.do({instruments_melody = instruments_melody.add(instruments[scale_steps[i]])})});

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
				\atk, 0.08,
				\dec, 0.5,
				\sus, 0,
				\rel, 0.5,
				\buf, buf[0].bufnum,
				\numBufs, buf.size,
				\bufPos, 0.9,
				\pan, Pseq(melody_pans),
				\fx, ~vbus,
				\fxsend, -25,
			).play;
		};



		if(scene_buffer != nil){
			if(scene_buffer.numChannels == 1) {
				Synth.new(\audioFileMono, scene_args);
			} {
				Synth.new(\audioFileStereo, scene_args);
			};
		};


		Post << "Scale: " << scale.name << Char.nl;
		Post << "Root: " << root << Char.nl;
		Post << "Scene: " << scene_path << Char.nl;
		Post << "Steps: " << root << Char.nl;
		Post << "Octaves: " << octaves << Char.nl;
		Post << "scale_steps: " << scale_steps << Char.nl;
		Post << "Chords: " << chords << Char.nl;

		Post << "Things: " << amount_of_things << Char.nl;

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
		Post << "Pans: " << pans << Char.nl;
		Post << Char.nl;

		if(~output_file != nil){
			(durations.sum + 4).wait;
			0.exit;
		};
	});
})