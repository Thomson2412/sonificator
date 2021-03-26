(
// a SynthDef
SynthDef(\test, { | out, freq = 440, amp = 0.1, nharms = 10, pan = 0, gate = 1 |
	var audio = Shaper.ar(b, Blip.ar(freq, nharms), amp);
	// var audio = Blip.ar(freq, nharms, amp);
    var env = Linen.kr(gate, doneAction: Done.freeSelf);
    OffsetOut.ar(out, Pan2.ar(audio, pan, env) );
}).add;
)

(
SynthDef(\cfstring1, { arg i_out, freq = 360, gate = 1, pan, amp=0.1;
    var out, eg, fc, osc, a, b, w;
    fc = LinExp.kr(LFNoise1.kr(Rand(0.25, 0.4)), -1, 1, 500, 2000);
	osc = Mix.fill(8, {LFSaw.ar(freq * [Rand(0.99, 1.01), Rand(0.99, 1.01)], 0, amp)}).distort * 0.2;
    eg = EnvGen.kr(Env.asr(1, 1, 1), gate, doneAction: Done.freeSelf);
    out = eg * RLPF.ar(osc, fc, 0.1);
    #a, b = out;
    Out.ar(i_out, Mix.ar(PanAz.ar(4, [a, b], [pan, pan+0.3])));
}).add;
)

(
"https://github.com/SCLOrkHub/SCLOrkSynths/blob/master/SynthDefs/keyboards/FMRhodes1.scd".postln;
SynthDef(\FMRhodes1, {
    arg
    // standard meanings
    out = 0, freq = 440, gate = 1, pan = 0, amp = 0.1, att = 0.001, rel = 1, lfoSpeed = 4.8, inputLevel = 0.2,
    // all of these range from 0 to 1
    modIndex = 0.2, mix = 0.2, lfoDepth = 0.1;

    var env1, env2, env3, env4;
    var osc1, osc2, osc3, osc4, snd;

    env1 = Env.perc(att, rel * 1.25, inputLevel, curve: \lin).kr;
    env2 = Env.perc(att, rel, inputLevel, curve: \lin).kr;
    env3 = Env.perc(att, rel * 1.5, inputLevel, curve: \lin).kr;
    env4 = Env.perc(att, rel * 1.5, inputLevel, curve: \lin).kr;

    osc4 = SinOsc.ar(freq) * 6.7341546494171 * modIndex * env4;
    osc3 = SinOsc.ar(freq * 2, osc4) * env3;
    osc2 = SinOsc.ar(freq * 30) * 0.683729941 * env2;
    osc1 = SinOsc.ar(freq * 2, osc2) * env1;
    snd = Mix((osc3 * (1 - mix)) + (osc1 * mix));
  	snd = snd * (SinOsc.ar(lfoSpeed).range((1 - lfoDepth), 1));

    snd = snd * Env.asr(0, 1, 0.1).kr(gate: gate, doneAction: 2);
    snd = Pan2.ar(snd, pan, amp);

    Out.ar(out, snd);
}).add;
)
(
"https://github.com/SCLOrkHub/SCLOrkSynths/blob/master/SynthDefs/keyboards/FMRhodes2.scd".postln;
SynthDef(\FMRhodes2, {
    arg
    // standard meanings
    out = 0, freq = 440, gate = 1, pan = 0, amp = 0.1, att = 0.001, rel = 1,
	//controls
	lfoSpeed = 2.4, inputLevel = 0.2, modIndex = 2, fmHarmonic = 30,
    // all of these range from 0 to 1
    mix = 0.2, lfoDepth = 0.75, lfoBottom = 0.15, lfoRange = 0.3;

    var env1, env2, env3, env4, range, bottom;
    var osc1, osc2, osc3, osc4, snd;

	bottom = lfoBottom.linlin(0, 1, 100, 12000);
	range = lfoRange.linlin(0, 1, bottom, 16000);

    env1 = Env.perc(att, rel * 1.25, inputLevel, curve: \lin).kr;
    env2 = Env.perc(att, rel, inputLevel, curve: \lin).kr;
    env3 = Env.perc(att, rel * 1.5, inputLevel, curve: \lin).kr;
    env4 = Env.perc(att, rel * 1.5, inputLevel, curve: \lin).kr;

    osc4 = SinOsc.ar(freq) * 6.7341546494171 * modIndex * env4;
    osc3 = SinOsc.ar(freq * 2, osc4) * env3;
    osc2 = SinOsc.ar(freq * fmHarmonic) * 0.683729941 * env2;
    osc1 = SinOsc.ar(freq * 2, osc2) * env1;

    snd = Mix((osc3 * (1 - mix)) + (osc1 * mix));
	snd = LPF.ar(snd, SinOsc.kr(lfoSpeed).range(bottom, range), lfoDepth) + (snd * (1 - lfoDepth));
    snd = snd * Env.asr(0, 1, 0.1).kr(gate: gate, doneAction: 2);
    snd = Pan2.ar(snd, pan, amp);

    Out.ar(out, snd);
}).add;
)
(
"https://github.com/SCLOrkHub/SCLOrkSynths/blob/master/SynthDefs/keyboards/everythingRhodes.scd".postln;
SynthDef("everythingRhodes", {
	arg
	//Standard Definitions
	out = 0, freq = 440, amp = 0.1, att = 0.001, rel = 3, pan = 0,
	//Other Controls
	filterRange = 2000, rq = 0.5, harmonic = 2.pow(33.5/12),
	//These Range from 0 to 1
	width = 0.2, mix = 0.7;

    //Electric Piano
	var snd, env, lfo;

	env = Env.perc(att, rel).kr(doneAction: 2); //Original used an ADSR envelope.

	snd = Pulse.ar(
	        	freq: freq * [1, harmonic], //Original used 33.5.midiratio here, which is just below the 7th harmonic.
	        	width: [width, width/2],
		        mul: [mix, (1 - mix)]
            );
	snd  = BLowPass4.ar(
		        in: snd,
		        freq: (filterRange * env) + (2 * freq),
		        rq: rq
		    );
    snd = Mix.ar(snd) * env * amp;

	Out.ar(out, Pan2.ar(snd, pan));
},
metadata: (
	credit: "Nick Collins, from Mitchell Sigman, 2011. http://www.sussex.ac.uk/Users/nc81/index.html",
	category: \keyboards,
	tags: [\rhodes, \piano, \pitched]
	)
).add
)


(
var scale;
var amp;
var duration;
var release;
var strum;
var scale_notes = Pseq([[0,2,4,6]]);
var line;
b.free;
b = Buffer.alloc(s, 80, 2);
o.free;
o = OSCFunc(
	{
		arg msg, time, addr, recvPort;
		msg.postln;
		scale = Scale.major;
		if(msg[3] < 4,{
			scale = Scale.minor;
		},);
		amp = msg[2] / 1000;
		duration = msg[4] / 1000;
		release = duration * 0.5;
		strum = 0.08.rand + 0.02;
		line = (msg[5] - 50) / 10;

		// t = Signal.fill(line.size, {|i| line[i]});
		t = Signal.chebyFill(line.size, line);
		b.sendCollection(t.asWavetableNoWrap);

		Pbind(
			\instrument, \test,
			\degree, scale_notes,
			\scale, scale,
			\root, msg[1],
			\octave, msg[3],
			\dur, duration,
			\strum, strum,
			\amp, amp,
			\att, 0.1,
			\rel, release,
		).play;
		Post <<< "Scale: " << scale.name << Char.nl;
		Post <<< "Root: " << msg[1] << Char.nl;
		Post <<< "Octave: " << msg[3] << Char.nl;
		Post <<< "Amp: " << amp << Char.nl;
		Post <<< "Duration: " << duration << Char.nl;
		Post <<< "Release: " << release << Char.nl;
		Post <<< "Strum: " << strum << Char.nl;
		Post <<< "Scale notes: " << scale_notes << Char.nl;
		Post <<< "Line: " << line << Char.nl;
		Post << Char.nl;
	},
	'/low_level_data', nil, 8484);
)
b.plot;

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

b = Buffer.alloc(s, l.size * 2, 1);
l.plot;
t = Signal.fill(l.size, {|i| l[i]});
t = Signal.chebyFill(l.size, l);

t.plot
// t.asWavetableNoWrap will convert it to the official Wavetable format at next power of two size
b.sendCollection(t.asWavetableNoWrap);  // may also use loadCollection here
b.plot

(
{
    Shaper.ar(
        b,
		SinOsc.ar([440, 440], 0.5)
    )
}.scope
)

b.free;

Wavetable.chebyFill(513, [1]).plot;

// shifted to avoid DC offset when waveshaping a zero signal
Wavetable.chebyFill(513, [0, 1], zeroOffset: true).plot;

// normalized sum of (unshifted) Chebyshev polynomials (the default)
Wavetable.chebyFill(513, [0, 1, 0, 0, 0, 1], normalize: true, zeroOffset: false).plot;

Wavetable.chebyFill(513, [0, 0, 1]).plot;
Wavetable.chebyFill(513, [0.3, -0.8, 1.1]).plot;
l[0] = 2
l[l.size-1] = -2
Wavetable.chebyFill(61, l).plot;

(
s.waitForBoot({
    var amplitudes = [0, 1, 1, -2, 1];
    var wavs = [
        Wavetable.chebyFill(256+1, amplitudes, normalize: true, zeroOffset: true),
        Wavetable.chebyFill(256+1, amplitudes, normalize: true, zeroOffset: false)
    ];
    b = wavs.collect{ arg wav; Buffer.loadCollection(s, wav) };
    s.sync;
    x = {
        var in = SinOsc.ar(100, 0);
        Shaper.ar(b, in ) ++ LeakDC.ar(Shaper.ar(b[1], in))
    }.scope;
})
)
x.free; b.do(_.free); b = nil;
Wavetable.sineFill(512, [0.5]).plot;
Wavetable.sineFill(512, [1]).asSignal.plot;

(
Pbind(
	\instrument, \cfstring1,
	\degree, scale_notes,
    \scale, Scale.major,
	\root, 0,
	\octave, 4,
	\dur, 3,
	\strum, 0.08,
	\amp, 0.2
).play;
)

Pbind(
	\instrument, \FMRhodes1,
	\degree, scale_notes,
	\scale, scale,
	\root, msg[1],
	\octave, msg[3],
	\dur, duration,
	\strum, strum,
	\amp, amp,
	\att, 0.1,
	\rel, release,
).play;
Pbind(
	\instrument, \FMRhodes2,
	\degree, scale_notes,
	\scale, scale,
	\root, msg[1],
	\octave, msg[3],
	\dur, duration,
	\strum, strum,
	\amp, amp,
	\att, 0.1,
	\rel, release,
).play;
Pbind(
	\instrument, \everythingRhodes,
	\degree, scale_notes,
	\scale, scale,
	\root, msg[1],
	\octave, msg[3],
	\dur, duration,
	\strum, strum,
	\amp, amp / 5,
	\rel, release,
).play;


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

