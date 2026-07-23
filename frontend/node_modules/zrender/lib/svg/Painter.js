import { brush, setClipPath, setGradient, setPattern } from './graphic.js';
import { createElement, createVNode, vNodeToString, getCssString, createBrushScope, createSVGVNode } from './core.js';
import { normalizeColor, encodeBase64, isGradient, isPattern } from './helper.js';
import { extend, keys, logError, map, noop, retrieve2 } from '../core/util.js';
import patch, { updateAttrs } from './patch.js';
import { getSize } from '../canvas/helper.js';
var svgId = 0;
var SVGPainter = (function () {
    function SVGPainter(root, storage, opts) {
        this.type = 'svg';
        this.refreshHover = createMethodNotSupport('refreshHover');
        this.configLayer = createMethodNotSupport('configLayer');
        this.storage = storage;
        this._opts = opts = extend({}, opts);
        this.root = root;
        this._id = 'zr' + svgId++;
        this._oldVNode = createSVGVNode(opts.width, opts.height);
        if (root && !opts.ssr) {
            var viewport = this._viewport = document.createElement('div');
            viewport.style.cssText = 'position:relative;overflow:hidden';
            var svgDom = this._svgDom = this._oldVNode.elm = createElement('svg');
            updateAttrs(null, this._oldVNode);
            viewport.appendChild(svgDom);
            root.appendChild(viewport);
        }
        this.resize(opts.width, opts.height);
    }
    SVGPainter.prototype.getType = function () {
        return this.type;
    };
    SVGPainter.prototype.getViewportRoot = function () {
        return this._viewport;
    };
    SVGPainter.prototype.getViewportRootOffset = function () {
        var viewportRoot = this.getViewportRoot();
        if (viewportRoot) {
            return {
                offsetLeft: viewportRoot.offsetLeft || 0,
                offsetTop: viewportRoot.offsetTop || 0
            };
        }
    };
    SVGPainter.prototype.getSvgDom = function () {
        return this._svgDom;
    };
    SVGPainter.prototype.refresh = function () {
        if (this.root) {
            var vnode = this.renderToVNode({
                willUpdate: true
            });
            vnode.attrs.style = 'position:absolute;left:0;top:0;user-select:none';
            patch(this._oldVNode, vnode);
            this._oldVNode = vnode;
        }
    };
    SVGPainter.prototype.renderOneToVNode = function (el) {
        return brush(el, createBrushScope(this._id));
    };
    SVGPainter.prototype.renderToVNode = function (opts) {
        opts = opts || {};
        var list = this.storage.getDisplayList(true);
        var width = this._width;
        var height = this._height;
        var scope = createBrushScope(this._id);
        scope.animation = opts.animation;
        scope.willUpdate = opts.willUpdate;
        scope.compress = opts.compress;
        scope.emphasis = opts.emphasis;
        scope.ssr = this._opts.ssr;
        var children = [];
        var bgVNode = this._bgVNode = createBackgroundVNode(width, height, this._backgroundColor, scope);
        bgVNode && children.push(bgVNode);
        var mainVNode = !opts.compress
            ? (this._mainVNode = createVNode('g', 'main', {}, [])) : null;
        this._paintList(list, scope, mainVNode ? mainVNode.children : children);
        mainVNode && children.push(mainVNode);
        var defs = map(keys(scope.defs), function (id) { return scope.defs[id]; });
        if (defs.length) {
            children.push(createVNode('defs', 'defs', {}, defs));
        }
        if (opts.animation) {
            var animationCssStr = getCssString(scope.cssNodes, scope.cssAnims, { newline: true });
            if (animationCssStr) {
                var styleNode = createVNode('style', 'stl', {}, [], animationCssStr);
                children.push(styleNode);
            }
        }
        return createSVGVNode(width, height, children, opts.useViewBox);
    };
    SVGPainter.prototype.renderToString = function (opts) {
        opts = opts || {};
        return vNodeToString(this.renderToVNode({
            animation: retrieve2(opts.cssAnimation, true),
            emphasis: retrieve2(opts.cssEmphasis, true),
            willUpdate: false,
            compress: true,
            useViewBox: retrieve2(opts.useViewBox, true)
        }), { newline: true });
    };
    SVGPainter.prototype.setBackgroundColor = function (backgroundColor) {
        this._backgroundColor = backgroundColor;
    };
    SVGPainter.prototype.getSvgRoot = function () {
        return this._mainVNode && this._mainVNode.elm;
    };
    SVGPainter.prototype._paintList = function (list, scope, out) {
        var listLen = list.length;
        var clipPathsGroupsStack = [];
        var clipPathsGroupsStackDepth = 0;
        var currentClipPathGroup;
        var prevClipPaths;
        var clipGroupNodeIdx = 0;
        for (var i = 0; i < listLen; i++) {
            var displayable = list[i];
            if (!displayable.invisible) {
                var clipPaths = displayable.__clipPaths;
                var len = clipPaths && clipPaths.length || 0;
                var prevLen = prevClipPaths && prevClipPaths.length || 0;
                var lca = void 0;
                for (lca = Math.max(len - 1, prevLen - 1); lca >= 0; lca--) {
                    if (clipPaths && prevClipPaths
                        && clipPaths[lca] === prevClipPaths[lca]) {
                        break;
                    }
                }
                for (var i_1 = prevLen - 1; i_1 > lca; i_1--) {
                    clipPathsGroupsStackDepth--;
                    currentClipPathGroup = clipPathsGroupsStack[clipPathsGroupsStackDepth - 1];
                }
                for (var i_2 = lca + 1; i_2 < len; i_2++) {
                    var groupAttrs = {};
                    setClipPath(clipPaths[i_2], groupAttrs, scope);
                    var g = createVNode('g', 'clip-g-' + clipGroupNodeIdx++, groupAttrs, []);
                    (currentClipPathGroup ? currentClipPathGroup.children : out).push(g);
                    clipPathsGroupsStack[clipPathsGroupsStackDepth++] = g;
                    currentClipPathGroup = g;
                }
                prevClipPaths = clipPaths;
                var ret = brush(displayable, scope);
                if (ret) {
                    (currentClipPathGroup ? currentClipPathGroup.children : out).push(ret);
                }
            }
        }
    };
    SVGPainter.prototype.resize = function (width, height) {
        var opts = this._opts;
        var root = this.root;
        var viewport = this._viewport;
        width != null && (opts.width = width);
        height != null && (opts.height = height);
        if (root && viewport) {
            viewport.style.display = 'none';
            width = getSize(root, 0, opts);
            height = getSize(root, 1, opts);
            viewport.style.display = '';
        }
        if (this._width !== width || this._height !== height) {
            this._width = width;
            this._height = height;
            if (viewport) {
                var viewportStyle = viewport.style;
                viewportStyle.width = width + 'px';
                viewportStyle.height = height + 'px';
            }
            if (!isPattern(this._backgroundColor)) {
                var svgDom = this._svgDom;
                if (svgDom) {
                    svgDom.setAttribute('width', width);
                    svgDom.setAttribute('height', height);
                }
                var bgEl = this._bgVNode && this._bgVNode.elm;
                if (bgEl) {
                    bgEl.setAttribute('width', width);
                    bgEl.setAttribute('height', height);
                }
            }
            else {
                this.refresh();
            }
        }
    };
    SVGPainter.prototype.getWidth = function () {
        return this._width;
    };
    SVGPainter.prototype.getHeight = function () {
        return this._height;
    };
    SVGPainter.prototype.dispose = function () {
        if (this.root) {
            this.root.innerHTML = '';
        }
        this._svgDom =
            this._viewport =
                this.storage =
                    this._oldVNode =
                        this._bgVNode =
                            this._mainVNode = null;
    };
    SVGPainter.prototype.clear = function () {
        if (this._svgDom) {
            this._svgDom.innerHTML = null;
        }
        this._oldVNode = null;
    };
    SVGPainter.prototype.toDataURL = function (base64) {
        var str = this.renderToString();
        var prefix = 'data:image/svg+xml;';
        if (base64) {
            str = encodeBase64(str);
            return str && prefix + 'base64,' + str;
        }
        return prefix + 'charset=UTF-8,' + encodeURIComponent(str);
    };
    return SVGPainter;
}());
function createMethodNotSupport(method) {
    return function () {
        if (process.env.NODE_ENV !== 'production') {
            logError('In SVG mode painter not support method "' + method + '"');
        }
    };
}
function createBackgroundVNode(width, height, backgroundColor, scope) {
    var bgVNode;
    if (backgroundColor && backgroundColor !== 'none') {
        bgVNode = createVNode('rect', 'bg', {
            width: width,
            height: height,
            x: '0',
            y: '0'
        });
        if (isGradient(backgroundColor)) {
            setGradient({ fill: backgroundColor }, bgVNode.attrs, 'fill', scope);
        }
        else if (isPattern(backgroundColor)) {
            setPattern({
                style: {
                    fill: backgroundColor
                },
                dirty: noop,
                getBoundingRect: function () { return ({ width: width, height: height }); }
            }, bgVNode.attrs, 'fill', scope);
        }
        else {
            var _a = normalizeColor(backgroundColor), color = _a.color, opacity = _a.opacity;
            bgVNode.attrs.fill = color;
            opacity < 1 && (bgVNode.attrs['fill-opacity'] = opacity);
        }
    }
    return bgVNode;
}
export default SVGPainter;
