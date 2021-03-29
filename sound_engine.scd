(
~wt = Array.newClear(4);
~sine = Signal.sineFill(1024, [1, 0]).asWavetable;
~tri = Env([0, 1 , -1, 0], [1, 2, 1], \lin).asSignal(1024).asWavetable;
~saw = Env([0, 1 , -1, 0], [1, 0, 1], \lin).asSignal(1024).asWavetable;
~square = Signal.rectWindow(1024, 512).asWavetable;
~wt[0] = ~sine;
~wt[1] = ~tri;
~wt[2] = ~saw;
~wt[3] = ~square;
)
(
~sine.plot("~sine");
~tri.plot("~tri");
~saw.plot("~saw");
~square.plot("~square");
)


~buf = Buffer.allocConsecutive(4, s, 2048);
~buf.do({arg buf, i; buf.loadCollection(~wt[i])});


~wt.do({arg wtsig, i; wtsig.plot(~buf[i].bufnum.asString)})

(
SynthDef.new(\vosc, {
	arg buf=0, numBufs=2, freq=440, amp=0.2;
	var sig, bufPos, detuneSig;
	detuneSig = LFNoise1.kr(0.2!8).bipolar(0.2).midiratio;
	bufPos = buf + LFNoise1.kr(0.5).range(0, numBufs - 1);
	sig = VOsc.ar(bufPos, freq * detuneSig);
	sig = Splay.ar(sig);
	sig = LeakDC.ar(sig) * amp;
	Out.ar(0, sig);
}).add;
)

Synth(\vosc, [\freq, 60.midicps, \buf, ~buf[0].bufnum, \numBufs, 4]);


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

