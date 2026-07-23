import env from '../core/env.js';
var requestAnimationFrame;
requestAnimationFrame = (env.hasGlobalWindow
    && ((window.requestAnimationFrame && window.requestAnimationFrame.bind(window))
        || (window.msRequestAnimationFrame && window.msRequestAnimationFrame.bind(window))
        || window.mozRequestAnimationFrame
        || window.webkitRequestAnimationFrame)) || function (func) {
    return setTimeout(func, 16);
};
export default requestAnimationFrame;
