import env from './env.js';
import { buildTransformer } from './fourPointsTransform.js';
var EVENT_SAVED_PROP = '___zrEVENTSAVED';
var _calcOut = [];
export function transformLocalCoord(out, elFrom, elTarget, inX, inY) {
    return transformCoordWithViewport(_calcOut, elFrom, inX, inY, true)
        && transformCoordWithViewport(out, elTarget, _calcOut[0], _calcOut[1]);
}
export function transformCoordWithViewport(out, el, inX, inY, inverse) {
    if (el.getBoundingClientRect && env.domSupported && !isCanvasEl(el)) {
        var saved = el[EVENT_SAVED_PROP] || (el[EVENT_SAVED_PROP] = {});
        var markers = prepareCoordMarkers(el, saved);
        var transformer = preparePointerTransformer(markers, saved, inverse);
        if (transformer) {
            transformer(out, inX, inY);
            return true;
        }
    }
    return false;
}
function prepareCoordMarkers(el, saved) {
    var markers = saved.markers;
    if (markers) {
        return markers;
    }
    markers = saved.markers = [];
    var propLR = ['left', 'right'];
    var propTB = ['top', 'bottom'];
    for (var i = 0; i < 4; i++) {
        var marker = document.createElement('div');
        var stl = marker.style;
        var idxLR = i % 2;
        var idxTB = (i >> 1) % 2;
        stl.cssText = [
            'position: absolute',
            'visibility: hidden',
            'padding: 0',
            'margin: 0',
            'border-width: 0',
            'user-select: none',
            'width:0',
            'height:0',
            propLR[idxLR] + ':0',
            propTB[idxTB] + ':0',
            propLR[1 - idxLR] + ':auto',
            propTB[1 - idxTB] + ':auto',
            ''
        ].join('!important;');
        el.appendChild(marker);
        markers.push(marker);
    }
    return markers;
}
function preparePointerTransformer(markers, saved, inverse) {
    var transformerName = inverse ? 'invTrans' : 'trans';
    var transformer = saved[transformerName];
    var oldSrcCoords = saved.srcCoords;
    var srcCoords = [];
    var destCoords = [];
    var oldCoordTheSame = true;
    for (var i = 0; i < 4; i++) {
        var rect = markers[i].getBoundingClientRect();
        var ii = 2 * i;
        var x = rect.left;
        var y = rect.top;
        srcCoords.push(x, y);
        oldCoordTheSame = oldCoordTheSame && oldSrcCoords && x === oldSrcCoords[ii] && y === oldSrcCoords[ii + 1];
        destCoords.push(markers[i].offsetLeft, markers[i].offsetTop);
    }
    return (oldCoordTheSame && transformer)
        ? transformer
        : (saved.srcCoords = srcCoords,
            saved[transformerName] = inverse
                ? buildTransformer(destCoords, srcCoords)
                : buildTransformer(srcCoords, destCoords));
}
export function isCanvasEl(el) {
    return el.nodeName.toUpperCase() === 'CANVAS';
}
var replaceReg = /([&<>"'])/g;
var replaceMap = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    '\'': '&#39;'
};
export function encodeHTML(source) {
    return source == null
        ? ''
        : (source + '').replace(replaceReg, function (str, c) {
            return replaceMap[c];
        });
}
