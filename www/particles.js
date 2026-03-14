/**
 * Wind particle animation module.
 * Loaded automatically by trame via server.enable_module().
 * Controlled via trame state: particle_active, particle_count
 */
(function () {
  "use strict";

  // === CONFIG ===
  var MAX_AGE = 80;
  var FADE_ALPHA = 0.95;
  var LINE_WIDTH = 3.0;
  var LINE_ALPHA = 0.6;
  var SPEED_MAX_PX = 1.5;
  var MAX_WIND = 35;

  // === STATE ===
  var canvas, ctx, trailCanvas, trailCtx;
  var particles = [];
  var windField = null;
  var mapState = null;
  var animating = false;
  var moveTimer = null;
  var animFrame = null;
  var container = null;
  var numParticles = 1500;
  var fpsFrameCount = 0;
  var fpsLastTime = performance.now();
  var interactionEndTime = null;
  var measureRespawn = false;

  // === INIT ===
  function init() {
    container = document.querySelector(".leaflet-container");
    if (!container) {
      setTimeout(init, 500);
      return;
    }

    // Wait for tiles to load before fetching wind field (828 KB)
    // so we don't compete for HTTP connections during initial tile load
    setTimeout(function () {
      fetch("__wind/wind_field.json")
        .then(function (r) {
          return r.json();
        })
        .then(function (data) {
          windField = data;
          createCanvas();
          attachListeners();
          startAnimation();
        });
    }, 2000);
  }

  function createCanvas() {
    var dpr = window.devicePixelRatio || 1;
    var w = container.clientWidth;
    var h = container.clientHeight;

    canvas = document.createElement("canvas");
    canvas.id = "wind-particles";
    canvas.style.cssText =
      "position:absolute;top:0;left:0;pointer-events:none;z-index:450;";
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    container.appendChild(canvas);
    ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);

    trailCanvas = document.createElement("canvas");
    trailCanvas.width = w * dpr;
    trailCanvas.height = h * dpr;
    trailCtx = trailCanvas.getContext("2d");
    trailCtx.scale(dpr, dpr);
  }

  function resizeCanvas() {
    if (!canvas || !container) return;
    var dpr = window.devicePixelRatio || 1;
    var w = container.clientWidth;
    var h = container.clientHeight;
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);
    trailCanvas.width = w * dpr;
    trailCanvas.height = h * dpr;
    trailCtx.scale(dpr, dpr);
  }

  // === LISTENERS ===
  function attachListeners() {
    container.addEventListener("mousedown", onMoveStart, true);
    container.addEventListener("touchstart", onMoveStart, true);
    container.addEventListener(
      "wheel",
      function () {
        onMoveStart();
        clearTimeout(moveTimer);
        moveTimer = setTimeout(onMoveEnd, 600);
      },
      true
    );
    document.addEventListener("mouseup", onMoveEnd, true);
    document.addEventListener("touchend", onMoveEnd, true);
    window.addEventListener("resize", function () {
      resizeCanvas();
      onMoveStart();
      clearTimeout(moveTimer);
      moveTimer = setTimeout(onMoveEnd, 500);
    });
  }

  function onMoveStart() {
    animating = false;
    if (animFrame) {
      cancelAnimationFrame(animFrame);
      animFrame = null;
    }
    if (ctx && container) {
      var w = container.clientWidth;
      var h = container.clientHeight;
      ctx.clearRect(0, 0, w, h);
      trailCtx.clearRect(0, 0, w, h);
    }
  }

  function onMoveEnd() {
    clearTimeout(moveTimer);
    interactionEndTime = performance.now();
    measureRespawn = true;
    moveTimer = setTimeout(function () {
      startAnimation();
    }, 400);
  }

  // === MAP PROJECTION ===
  function updateMapState() {
    var tiles = container.querySelectorAll(".leaflet-tile");
    var cr = container.getBoundingClientRect();
    for (var i = 0; i < tiles.length; i++) {
      var src = tiles[i].src || "";
      var m = src.match(/\/tiles\/(\d+)\/(\d+)\/(\d+)\.png/);
      if (!m) continue;
      var rect = tiles[i].getBoundingClientRect();
      if (rect.width < 10) continue;
      mapState = {
        n: Math.pow(2, parseInt(m[1])),
        zoom: parseInt(m[1]),
        tw: rect.width,
        rtx: parseInt(m[2]),
        rty: parseInt(m[3]),
        rpx: rect.left - cr.left,
        rpy: rect.top - cr.top,
      };
      return true;
    }
    return false;
  }

  function pixelToLatLon(px, py) {
    var s = mapState;
    var tx = s.rtx + (px - s.rpx) / s.tw;
    var ty = s.rty + (py - s.rpy) / s.tw;
    var lon = (tx / s.n) * 360 - 180;
    var lat =
      Math.atan(Math.sinh(Math.PI * (1 - (2 * ty) / s.n))) * (180 / Math.PI);
    return [lat, lon];
  }

  // === WIND LOOKUP ===
  function getWind(lat, lon) {
    if (!windField) return [0, 0];
    while (lon > 180) lon -= 360;
    while (lon < -180) lon += 360;

    var li = (windField.lat_max - lat) / windField.lat_step;
    var lj = (lon - windField.lon_min) / windField.lon_step;
    var i0 = Math.floor(li),
      j0 = Math.floor(lj);
    var i1 = i0 + 1,
      j1 = j0 + 1;

    if (i0 < 0 || i1 >= windField.n_lats || j0 < 0 || j1 >= windField.n_lons)
      return [0, 0];

    var fi = li - i0,
      fj = lj - j0;
    var nl = windField.n_lons;
    var u = windField.u,
      v = windField.v;

    var ui =
      u[i0 * nl + j0] * (1 - fi) * (1 - fj) +
      u[i0 * nl + j1] * (1 - fi) * fj +
      u[i1 * nl + j0] * fi * (1 - fj) +
      u[i1 * nl + j1] * fi * fj;
    var vi =
      v[i0 * nl + j0] * (1 - fi) * (1 - fj) +
      v[i0 * nl + j1] * (1 - fi) * fj +
      v[i1 * nl + j0] * fi * (1 - fj) +
      v[i1 * nl + j1] * fi * fj;

    return [ui, vi];
  }

  // === PARTICLES ===
  function spawnParticle() {
    var w = container.clientWidth,
      h = container.clientHeight;
    return {
      x: Math.random() * w,
      y: Math.random() * h,
      age: Math.floor(Math.random() * MAX_AGE),
      maxAge: MAX_AGE + Math.floor(Math.random() * 20),
    };
  }

  function startAnimation() {
    if (!updateMapState()) {
      setTimeout(startAnimation, 500);
      return;
    }
    particles = [];
    for (var i = 0; i < numParticles; i++) particles.push(spawnParticle());

    var w = container.clientWidth,
      h = container.clientHeight;
    ctx.clearRect(0, 0, w, h);
    trailCtx.clearRect(0, 0, w, h);
    animating = true;
    animate();
  }

  function stopAnimation() {
    animating = false;
    if (animFrame) {
      cancelAnimationFrame(animFrame);
      animFrame = null;
    }
    if (ctx && container) {
      var w = container.clientWidth;
      var h = container.clientHeight;
      ctx.clearRect(0, 0, w, h);
      trailCtx.clearRect(0, 0, w, h);
    }
  }

  function animate() {
    if (!animating) return;

    if (measureRespawn && interactionEndTime) {
      var respawnMs = performance.now() - interactionEndTime;
      console.log("BENCHMARK: Respawn delay: " + respawnMs.toFixed(0) + "ms");
      measureRespawn = false;
    }

    var w = container.clientWidth,
      h = container.clientHeight;
    var speedScale =
      (SPEED_MAX_PX / MAX_WIND) * Math.pow(2, mapState.zoom - 3);

    // Fade trails
    trailCtx.clearRect(0, 0, w, h);
    trailCtx.drawImage(canvas, 0, 0, w, h);
    ctx.clearRect(0, 0, w, h);
    ctx.globalAlpha = FADE_ALPHA;
    ctx.drawImage(trailCanvas, 0, 0, w, h);
    ctx.globalAlpha = 1.0;

    // Draw new segments
    ctx.beginPath();
    ctx.strokeStyle = "rgba(255,255,255," + LINE_ALPHA + ")";
    ctx.lineWidth = LINE_WIDTH;

    for (var i = 0; i < particles.length; i++) {
      var p = particles[i];
      var ll = pixelToLatLon(p.x, p.y);
      if (!ll) {
        particles[i] = spawnParticle();
        continue;
      }

      var wind = getWind(ll[0], ll[1]);
      var dx = wind[0] * speedScale;
      var dy = -wind[1] * speedScale;

      if (Math.abs(dx) < 0.01 && Math.abs(dy) < 0.01) {
        p.age++;
      } else {
        var nx = p.x + dx,
          ny = p.y + dy;
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(nx, ny);
        p.x = nx;
        p.y = ny;
      }

      p.age++;
      if (
        p.age > p.maxAge ||
        p.x < -10 ||
        p.x > w + 10 ||
        p.y < -10 ||
        p.y > h + 10
      ) {
        particles[i] = spawnParticle();
      }
    }
    ctx.stroke();

    // FPS measurement
    fpsFrameCount++;
    var now = performance.now();
    if (now - fpsLastTime >= 2000) {
      var fps = (fpsFrameCount * 1000) / (now - fpsLastTime);
      console.log("BENCHMARK: Animation FPS: " + fps.toFixed(1));
      fpsFrameCount = 0;
      fpsLastTime = now;
    }

    animFrame = requestAnimationFrame(animate);
  }

  // === PUBLIC API (controlled via trame ClientStateChange) ===
  window._windParticles = {
    start: function () {
      if (!windField) return;
      startAnimation();
    },
    stop: stopAnimation,
    setCount: function (n) {
      numParticles = n;
    },
    isReady: function () {
      return !!windField;
    },
  };

  // Start
  init();
})();
