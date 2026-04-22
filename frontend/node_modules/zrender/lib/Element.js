import Transformable, { TRANSFORMABLE_PROPS } from './core/Transformable.js';
import Animator, { cloneValue } from './animation/Animator.js';
import BoundingRect from './core/BoundingRect.js';
import Eventful from './core/Eventful.js';
import { calculateTextPosition, parsePercent } from './contain/text.js';
import { guid, isObject, keys, extend, indexOf, logError, mixin, isArrayLike, isTypedArray, isGradientObject, filter, reduce } from './core/util.js';
import { LIGHT_LABEL_COLOR, DARK_LABEL_COLOR } from './config.js';
import { parse, stringify } from './tool/color.js';
import { REDRAW_BIT } from './graphic/constants.js';
export var PRESERVED_NORMAL_STATE = '__zr_normal__';
var PRIMARY_STATES_KEYS = TRANSFORMABLE_PROPS.concat(['ignore']);
var DEFAULT_ANIMATABLE_MAP = reduce(TRANSFORMABLE_PROPS, function (obj, key) {
    obj[key] = true;
    return obj;
}, { ignore: false });
var tmpTextPosCalcRes = {};
var tmpBoundingRect = new BoundingRect(0, 0, 0, 0);
var Element = (function () {
    function Element(props) {
        this.id = guid();
        this.animators = [];
        this.currentStates = [];
        this.states = {};
        this._init(props);
    }
    Element.prototype._init = function (props) {
        this.attr(props);
    };
    Element.prototype.drift = function (dx, dy, e) {
        switch (this.draggable) {
            case 'horizontal':
                dy = 0;
                break;
            case 'vertical':
                dx = 0;
                break;
        }
        var m = this.transform;
        if (!m) {
            m = this.transform = [1, 0, 0, 1, 0, 0];
        }
        m[4] += dx;
        m[5] += dy;
        this.decomposeTransform();
        this.markRedraw();
    };
    Element.prototype.beforeUpdate = function () { };
    Element.prototype.afterUpdate = function () { };
    Element.prototype.update = function () {
        this.updateTransform();
        if (this.__dirty) {
            this.updateInnerText();
        }
    };
    Element.prototype.updateInnerText = function (forceUpdate) {
        var textEl = this._textContent;
        if (textEl && (!textEl.ignore || forceUpdate)) {
            if (!this.textConfig) {
                this.textConfig = {};
            }
            var textConfig = this.textConfig;
            var isLocal = textConfig.local;
            var innerTransformable = textEl.innerTransformable;
            var textAlign = void 0;
            var textVerticalAlign = void 0;
            var textStyleChanged = false;
            innerTransformable.parent = isLocal ? this : null;
            var innerOrigin = false;
            innerTransformable.copyTransform(textEl);
            if (textConfig.position != null) {
                var layoutRect = tmpBoundingRect;
                if (textConfig.layoutRect) {
                    layoutRect.copy(textConfig.layoutRect);
                }
                else {
                    layoutRect.copy(this.getBoundingRect());
                }
                if (!isLocal) {
                    layoutRect.applyTransform(this.transform);
                }
                if (this.calculateTextPosition) {
                    this.calculateTextPosition(tmpTextPosCalcRes, textConfig, layoutRect);
                }
                else {
                    calculateTextPosition(tmpTextPosCalcRes, textConfig, layoutRect);
                }
                innerTransformable.x = tmpTextPosCalcRes.x;
                innerTransformable.y = tmpTextPosCalcRes.y;
                textAlign = tmpTextPosCalcRes.align;
                textVerticalAlign = tmpTextPosCalcRes.verticalAlign;
                var textOrigin = textConfig.origin;
                if (textOrigin && textConfig.rotation != null) {
                    var relOriginX = void 0;
                    var relOriginY = void 0;
                    if (textOrigin === 'center') {
                        relOriginX = layoutRect.width * 0.5;
                        relOriginY = layoutRect.height * 0.5;
                    }
                    else {
                        relOriginX = parsePercent(textOrigin[0], layoutRect.width);
                        relOriginY = parsePercent(textOrigin[1], layoutRect.height);
                    }
                    innerOrigin = true;
                    innerTransformable.originX = -innerTransformable.x + relOriginX + (isLocal ? 0 : layoutRect.x);
                    innerTransformable.originY = -innerTransformable.y + relOriginY + (isLocal ? 0 : layoutRect.y);
                }
            }
            if (textConfig.rotation != null) {
                innerTransformable.rotation = textConfig.rotation;
            }
            var textOffset = textConfig.offset;
            if (textOffset) {
                innerTransformable.x += textOffset[0];
                innerTransformable.y += textOffset[1];
                if (!innerOrigin) {
                    innerTransformable.originX = -textOffset[0];
                    innerTransformable.originY = -textOffset[1];
                }
            }
            var isInside = textConfig.inside == null
                ? (typeof textConfig.position === 'string' && textConfig.position.indexOf('inside') >= 0)
                : textConfig.inside;
            var innerTextDefaultStyle = this._innerTextDefaultStyle || (this._innerTextDefaultStyle = {});
            var textFill = void 0;
            var textStroke = void 0;
            var autoStroke = void 0;
            if (isInside && this.canBeInsideText()) {
                textFill = textConfig.insideFill;
                textStroke = textConfig.insideStroke;
                if (textFill == null || textFill === 'auto') {
                    textFill = this.getInsideTextFill();
                }
                if (textStroke == null || textStroke === 'auto') {
                    textStroke = this.getInsideTextStroke(textFill);
                    autoStroke = true;
                }
            }
            else {
                textFill = textConfig.outsideFill;
                textStroke = textConfig.outsideStroke;
                if (textFill == null || textFill === 'auto') {
                    textFill = this.getOutsideFill();
                }
                if (textStroke == null || textStroke === 'auto') {
                    textStroke = this.getOutsideStroke(textFill);
                    autoStroke = true;
                }
            }
            textFill = textFill || '#000';
            if (textFill !== innerTextDefaultStyle.fill
                || textStroke !== innerTextDefaultStyle.stroke
                || autoStroke !== innerTextDefaultStyle.autoStroke
                || textAlign !== innerTextDefaultStyle.align
                || textVerticalAlign !== innerTextDefaultStyle.verticalAlign) {
                textStyleChanged = true;
                innerTextDefaultStyle.fill = textFill;
                innerTextDefaultStyle.stroke = textStroke;
                innerTextDefaultStyle.autoStroke = autoStroke;
                innerTextDefaultStyle.align = textAlign;
                innerTextDefaultStyle.verticalAlign = textVerticalAlign;
                textEl.setDefaultTextStyle(innerTextDefaultStyle);
            }
            textEl.__dirty |= REDRAW_BIT;
            if (textStyleChanged) {
                textEl.dirtyStyle(true);
            }
        }
    };
    Element.prototype.canBeInsideText = function () {
        return true;
    };
    Element.prototype.getInsideTextFill = function () {
        return '#fff';
    };
    Element.prototype.getInsideTextStroke = function (textFill) {
        return '#000';
    };
    Element.prototype.getOutsideFill = function () {
        return this.__zr && this.__zr.isDarkMode() ? LIGHT_LABEL_COLOR : DARK_LABEL_COLOR;
    };
    Element.prototype.getOutsideStroke = function (textFill) {
        var backgroundColor = this.__zr && this.__zr.getBackgroundColor();
        var colorArr = typeof backgroundColor === 'string' && parse(backgroundColor);
        if (!colorArr) {
            colorArr = [255, 255, 255, 1];
        }
        var alpha = colorArr[3];
        var isDark = this.__zr.isDarkMode();
        for (var i = 0; i < 3; i++) {
            colorArr[i] = colorArr[i] * alpha + (isDark ? 0 : 255) * (1 - alpha);
        }
        colorArr[3] = 1;
        return stringify(colorArr, 'rgba');
    };
    Element.prototype.traverse = function (cb, context) { };
    Element.prototype.attrKV = function (key, value) {
        if (key === 'textConfig') {
            this.setTextConfig(value);
        }
        else if (key === 'textContent') {
            this.setTextContent(value);
        }
        else if (key === 'clipPath') {
            this.setClipPath(value);
        }
        else if (key === 'extra') {
            this.extra = this.extra || {};
            extend(this.extra, value);
        }
        else {
            this[key] = value;
        }
    };
    Element.prototype.hide = function () {
        this.ignore = true;
        this.markRedraw();
    };
    Element.prototype.show = function () {
        this.ignore = false;
        this.markRedraw();
    };
    Element.prototype.attr = function (keyOrObj, value) {
        if (typeof keyOrObj === 'string') {
            this.attrKV(keyOrObj, value);
        }
        else if (isObject(keyOrObj)) {
            var obj = keyOrObj;
            var keysArr = keys(obj);
            for (var i = 0; i < keysArr.length; i++) {
                var key = keysArr[i];
                this.attrKV(key, keyOrObj[key]);
            }
        }
        this.markRedraw();
        return this;
    };
    Element.prototype.saveCurrentToNormalState = function (toState) {
        this._innerSaveToNormal(toState);
        var normalState = this._normalState;
        for (var i = 0; i < this.animators.length; i++) {
            var animator = this.animators[i];
            var fromStateTransition = animator.__fromStateTransition;
            if (animator.getLoop() || fromStateTransition && fromStateTransition !== PRESERVED_NORMAL_STATE) {
                continue;
            }
            var targetName = animator.targetName;
            var target = targetName
                ? normalState[targetName] : normalState;
            animator.saveTo(target);
        }
    };
    Element.prototype._innerSaveToNormal = function (toState) {
        var normalState = this._normalState;
        if (!normalState) {
            normalState = this._normalState = {};
        }
        if (toState.textConfig && !normalState.textConfig) {
            normalState.textConfig = this.textConfig;
        }
        this._savePrimaryToNormal(toState, normalState, PRIMARY_STATES_KEYS);
    };
    Element.prototype._savePrimaryToNormal = function (toState, normalState, primaryKeys) {
        for (var i = 0; i < primaryKeys.length; i++) {
            var key = primaryKeys[i];
            if (toState[key] != null && !(key in normalState)) {
                normalState[key] = this[key];
            }
        }
    };
    Element.prototype.hasState = function () {
        return this.currentStates.length > 0;
    };
    Element.prototype.getState = function (name) {
        return this.states[name];
    };
    Element.prototype.ensureState = function (name) {
        var states = this.states;
        if (!states[name]) {
            states[name] = {};
        }
        return states[name];
    };
    Element.prototype.clearStates = function (noAnimation) {
        this.useState(PRESERVED_NORMAL_STATE, false, noAnimation);
    };
    Element.prototype.useState = function (stateName, keepCurrentStates, noAnimation, forceUseHoverLayer) {
        var toNormalState = stateName === PRESERVED_NORMAL_STATE;
        var hasStates = this.hasState();
        if (!hasStates && toNormalState) {
            return;
        }
        var currentStates = this.currentStates;
        var animationCfg = this.stateTransition;
        if (indexOf(currentStates, stateName) >= 0 && (keepCurrentStates || currentStates.length === 1)) {
            return;
        }
        var state;
        if (this.stateProxy && !toNormalState) {
            state = this.stateProxy(stateName);
        }
        if (!state) {
            state = (this.states && this.states[stateName]);
        }
        if (!state && !toNormalState) {
            logError("State " + stateName + " not exists.");
            return;
        }
        if (!toNormalState) {
            this.saveCurrentToNormalState(state);
        }
        var useHoverLayer = !!((state && state.hoverLayer) || forceUseHoverLayer);
        if (useHoverLayer) {
            this._toggleHoverLayerFlag(true);
        }
        this._applyStateObj(stateName, state, this._normalState, keepCurrentStates, !noAnimation && !this.__inHover && animationCfg && animationCfg.duration > 0, animationCfg);
        var textContent = this._textContent;
        var textGuide = this._textGuide;
        if (textContent) {
            textContent.useState(stateName, keepCurrentStates, noAnimation, useHoverLayer);
        }
        if (textGuide) {
            textGuide.useState(stateName, keepCurrentStates, noAnimation, useHoverLayer);
        }
        if (toNormalState) {
            this.currentStates = [];
            this._normalState = {};
        }
        else {
            if (!keepCurrentStates) {
                this.currentStates = [stateName];
            }
            else {
                this.currentStates.push(stateName);
            }
        }
        this._updateAnimationTargets();
        this.markRedraw();
        if (!useHoverLayer && this.__inHover) {
            this._toggleHoverLayerFlag(false);
            this.__dirty &= ~REDRAW_BIT;
        }
        return state;
    };
    Element.prototype.useStates = function (states, noAnimation, forceUseHoverLayer) {
        if (!states.length) {
            this.clearStates();
        }
        else {
            var stateObjects = [];
            var currentStates = this.currentStates;
            var len = states.length;
            var notChange = len === currentStates.length;
            if (notChange) {
                for (var i = 0; i < len; i++) {
                    if (states[i] !== currentStates[i]) {
                        notChange = false;
                        break;
                    }
                }
            }
            if (notChange) {
                return;
            }
            for (var i = 0; i < len; i++) {
                var stateName = states[i];
                var stateObj = void 0;
                if (this.stateProxy) {
                    stateObj = this.stateProxy(stateName, states);
                }
                if (!stateObj) {
                    stateObj = this.states[stateName];
                }
                if (stateObj) {
                    stateObjects.push(stateObj);
                }
            }
            var lastStateObj = stateObjects[len - 1];
            var useHoverLayer = !!((lastStateObj && lastStateObj.hoverLayer) || forceUseHoverLayer);
            if (useHoverLayer) {
                this._toggleHoverLayerFlag(true);
            }
            var mergedState = this._mergeStates(stateObjects);
            var animationCfg = this.stateTransition;
            this.saveCurrentToNormalState(mergedState);
            this._applyStateObj(states.join(','), mergedState, this._normalState, false, !noAnimation && !this.__inHover && animationCfg && animationCfg.duration > 0, animationCfg);
            var textContent = this._textContent;
            var textGuide = this._textGuide;
            if (textContent) {
                textContent.useStates(states, noAnimation, useHoverLayer);
            }
            if (textGuide) {
                textGuide.useStates(states, noAnimation, useHoverLayer);
            }
            this._updateAnimationTargets();
            this.currentStates = states.slice();
            this.markRedraw();
            if (!useHoverLayer && this.__inHover) {
                this._toggleHoverLayerFlag(false);
                this.__dirty &= ~REDRAW_BIT;
            }
        }
    };
    Element.prototype.isSilent = function () {
        var isSilent = this.silent;
        var ancestor = this.parent;
        while (!isSilent && ancestor) {
            if (ancestor.silent) {
                isSilent = true;
                break;
            }
            ancestor = ancestor.parent;
        }
        return isSilent;
    };
    Element.prototype._updateAnimationTargets = function () {
        for (var i = 0; i < this.animators.length; i++) {
            var animator = this.animators[i];
            if (animator.targetName) {
                animator.changeTarget(this[animator.targetName]);
            }
        }
    };
    Element.prototype.removeState = function (state) {
        var idx = indexOf(this.currentStates, state);
        if (idx >= 0) {
            var currentStates = this.currentStates.slice();
            currentStates.splice(idx, 1);
            this.useStates(currentStates);
        }
    };
    Element.prototype.replaceState = function (oldState, newState, forceAdd) {
        var currentStates = this.currentStates.slice();
        var idx = indexOf(currentStates, oldState);
        var newStateExists = indexOf(currentStates, newState) >= 0;
        if (idx >= 0) {
            if (!newStateExists) {
                currentStates[idx] = newState;
            }
            else {
                currentStates.splice(idx, 1);
            }
        }
        else if (forceAdd && !newStateExists) {
            currentStates.push(newState);
        }
        this.useStates(currentStates);
    };
    Element.prototype.toggleState = function (state, enable) {
        if (enable) {
            this.useState(state, true);
        }
        else {
            this.removeState(state);
        }
    };
    Element.prototype._mergeStates = function (states) {
        var mergedState = {};
        var mergedTextConfig;
        for (var i = 0; i < states.length; i++) {
            var state = states[i];
            extend(mergedState, state);
            if (state.textConfig) {
                mergedTextConfig = mergedTextConfig || {};
                extend(mergedTextConfig, state.textConfig);
            }
        }
        if (mergedTextConfig) {
            mergedState.textConfig = mergedTextConfig;
        }
        return mergedState;
    };
    Element.prototype._applyStateObj = function (stateName, state, normalState, keepCurrentStates, transition, animationCfg) {
        var needsRestoreToNormal = !(state && keepCurrentStates);
        if (state && state.textConfig) {
            this.textConfig = extend({}, keepCurrentStates ? this.textConfig : normalState.textConfig);
            extend(this.textConfig, state.textConfig);
        }
        else if (needsRestoreToNormal) {
            if (normalState.textConfig) {
                this.textConfig = normalState.textConfig;
            }
        }
        var transitionTarget = {};
        var hasTransition = false;
        for (var i = 0; i < PRIMARY_STATES_KEYS.length; i++) {
            var key = PRIMARY_STATES_KEYS[i];
            var propNeedsTransition = transition && DEFAULT_ANIMATABLE_MAP[key];
            if (state && state[key] != null) {
                if (propNeedsTransition) {
                    hasTransition = true;
                    transitionTarget[key] = state[key];
                }
                else {
                    this[key] = state[key];
                }
            }
            else if (needsRestoreToNormal) {
                if (normalState[key] != null) {
                    if (propNeedsTransition) {
                        hasTransition = true;
                        transitionTarget[key] = normalState[key];
                    }
                    else {
                        this[key] = normalState[key];
                    }
                }
            }
        }
        if (!transition) {
            for (var i = 0; i < this.animators.length; i++) {
                var animator = this.animators[i];
                var targetName = animator.targetName;
                if (!animator.getLoop()) {
                    animator.__changeFinalValue(targetName
                        ? (state || normalState)[targetName]
                        : (state || normalState));
                }
            }
        }
        if (hasTransition) {
            this._transitionState(stateName, transitionTarget, animationCfg);
        }
    };
    Element.prototype._attachComponent = function (componentEl) {
        if (componentEl.__zr && !componentEl.__hostTarget) {
            if (process.env.NODE_ENV !== 'production') {
                throw new Error('Text element has been added to zrender.');
            }
            return;
        }
        if (componentEl === this) {
            if (process.env.NODE_ENV !== 'production') {
                throw new Error('Recursive component attachment.');
            }
            return;
        }
        var zr = this.__zr;
        if (zr) {
            componentEl.addSelfToZr(zr);
        }
        componentEl.__zr = zr;
        componentEl.__hostTarget = this;
    };
    Element.prototype._detachComponent = function (componentEl) {
        if (componentEl.__zr) {
            componentEl.removeSelfFromZr(componentEl.__zr);
        }
        componentEl.__zr = null;
        componentEl.__hostTarget = null;
    };
    Element.prototype.getClipPath = function () {
        return this._clipPath;
    };
    Element.prototype.setClipPath = function (clipPath) {
        if (this._clipPath && this._clipPath !== clipPath) {
            this.removeClipPath();
        }
        this._attachComponent(clipPath);
        this._clipPath = clipPath;
        this.markRedraw();
    };
    Element.prototype.removeClipPath = function () {
        var clipPath = this._clipPath;
        if (clipPath) {
            this._detachComponent(clipPath);
            this._clipPath = null;
            this.markRedraw();
        }
    };
    Element.prototype.getTextContent = function () {
        return this._textContent;
    };
    Element.prototype.setTextContent = function (textEl) {
        var previousTextContent = this._textContent;
        if (previousTextContent === textEl) {
            return;
        }
        if (previousTextContent && previousTextContent !== textEl) {
            this.removeTextContent();
        }
        if (process.env.NODE_ENV !== 'production') {
            if (textEl.__zr && !textEl.__hostTarget) {
                throw new Error('Text element has been added to zrender.');
            }
        }
        textEl.innerTransformable = new Transformable();
        this._attachComponent(textEl);
        this._textContent = textEl;
        this.markRedraw();
    };
    Element.prototype.setTextConfig = function (cfg) {
        if (!this.textConfig) {
            this.textConfig = {};
        }
        extend(this.textConfig, cfg);
        this.markRedraw();
    };
    Element.prototype.removeTextConfig = function () {
        this.textConfig = null;
        this.markRedraw();
    };
    Element.prototype.removeTextContent = function () {
        var textEl = this._textContent;
        if (textEl) {
            textEl.innerTransformable = null;
            this._detachComponent(textEl);
            this._textContent = null;
            this._innerTextDefaultStyle = null;
            this.markRedraw();
        }
    };
    Element.prototype.getTextGuideLine = function () {
        return this._textGuide;
    };
    Element.prototype.setTextGuideLine = function (guideLine) {
        if (this._textGuide && this._textGuide !== guideLine) {
            this.removeTextGuideLine();
        }
        this._attachComponent(guideLine);
        this._textGuide = guideLine;
        this.markRedraw();
    };
    Element.prototype.removeTextGuideLine = function () {
        var textGuide = this._textGuide;
        if (textGuide) {
            this._detachComponent(textGuide);
            this._textGuide = null;
            this.markRedraw();
        }
    };
    Element.prototype.markRedraw = function () {
        this.__dirty |= REDRAW_BIT;
        var zr = this.__zr;
        if (zr) {
            if (this.__inHover) {
                zr.refreshHover();
            }
            else {
                zr.refresh();
            }
        }
        if (this.__hostTarget) {
            this.__hostTarget.markRedraw();
        }
    };
    Element.prototype.dirty = function () {
        this.markRedraw();
    };
    Element.prototype._toggleHoverLayerFlag = function (inHover) {
        this.__inHover = inHover;
        var textContent = this._textContent;
        var textGuide = this._textGuide;
        if (textContent) {
            textContent.__inHover = inHover;
        }
        if (textGuide) {
            textGuide.__inHover = inHover;
        }
    };
    Element.prototype.addSelfToZr = function (zr) {
        if (this.__zr === zr) {
            return;
        }
        this.__zr = zr;
        var animators = this.animators;
        if (animators) {
            for (var i = 0; i < animators.length; i++) {
                zr.animation.addAnimator(animators[i]);
            }
        }
        if (this._clipPath) {
            this._clipPath.addSelfToZr(zr);
        }
        if (this._textContent) {
            this._textContent.addSelfToZr(zr);
        }
        if (this._textGuide) {
            this._textGuide.addSelfToZr(zr);
        }
    };
    Element.prototype.removeSelfFromZr = function (zr) {
        if (!this.__zr) {
            return;
        }
        this.__zr = null;
        var animators = this.animators;
        if (animators) {
            for (var i = 0; i < animators.length; i++) {
                zr.animation.removeAnimator(animators[i]);
            }
        }
        if (this._clipPath) {
            this._clipPath.removeSelfFromZr(zr);
        }
        if (this._textContent) {
            this._textContent.removeSelfFromZr(zr);
        }
        if (this._textGuide) {
            this._textGuide.removeSelfFromZr(zr);
        }
    };
    Element.prototype.animate = function (key, loop, allowDiscreteAnimation) {
        var target = key ? this[key] : this;
        if (process.env.NODE_ENV !== 'production') {
            if (!target) {
                logError('Property "'
                    + key
                    + '" is not existed in element '
                    + this.id);
                return;
            }
        }
        var animator = new Animator(target, loop, allowDiscreteAnimation);
        key && (animator.targetName = key);
        this.addAnimator(animator, key);
        return animator;
    };
    Element.prototype.addAnimator = function (animator, key) {
        var zr = this.__zr;
        var el = this;
        animator.during(function () {
            el.updateDuringAnimation(key);
        }).done(function () {
            var animators = el.animators;
            var idx = indexOf(animators, animator);
            if (idx >= 0) {
                animators.splice(idx, 1);
            }
        });
        this.animators.push(animator);
        if (zr) {
            zr.animation.addAnimator(animator);
        }
        zr && zr.wakeUp();
    };
    Element.prototype.updateDuringAnimation = function (key) {
        this.markRedraw();
    };
    Element.prototype.stopAnimation = function (scope, forwardToLast) {
        var animators = this.animators;
        var len = animators.length;
        var leftAnimators = [];
        for (var i = 0; i < len; i++) {
            var animator = animators[i];
            if (!scope || scope === animator.scope) {
                animator.stop(forwardToLast);
            }
            else {
                leftAnimators.push(animator);
            }
        }
        this.animators = leftAnimators;
        return this;
    };
    Element.prototype.animateTo = function (target, cfg, animationProps) {
        animateTo(this, target, cfg, animationProps);
    };
    Element.prototype.animateFrom = function (target, cfg, animationProps) {
        animateTo(this, target, cfg, animationProps, true);
    };
    Element.prototype._transitionState = function (stateName, target, cfg, animationProps) {
        var animators = animateTo(this, target, cfg, animationProps);
        for (var i = 0; i < animators.length; i++) {
            animators[i].__fromStateTransition = stateName;
        }
    };
    Element.prototype.getBoundingRect = function () {
        return null;
    };
    Element.prototype.getPaintRect = function () {
        return null;
    };
    Element.initDefaultProps = (function () {
        var elProto = Element.prototype;
        elProto.type = 'element';
        elProto.name = '';
        elProto.ignore =
            elProto.silent =
                elProto.isGroup =
                    elProto.draggable =
                        elProto.dragging =
                            elProto.ignoreClip =
                                elProto.__inHover = false;
        elProto.__dirty = REDRAW_BIT;
        var logs = {};
        function logDeprecatedError(key, xKey, yKey) {
            if (!logs[key + xKey + yKey]) {
                console.warn("DEPRECATED: '" + key + "' has been deprecated. use '" + xKey + "', '" + yKey + "' instead");
                logs[key + xKey + yKey] = true;
            }
        }
        function createLegacyProperty(key, privateKey, xKey, yKey) {
            Object.defineProperty(elProto, key, {
                get: function () {
                    if (process.env.NODE_ENV !== 'production') {
                        logDeprecatedError(key, xKey, yKey);
                    }
                    if (!this[privateKey]) {
                        var pos = this[privateKey] = [];
                        enhanceArray(this, pos);
                    }
                    return this[privateKey];
                },
                set: function (pos) {
                    if (process.env.NODE_ENV !== 'production') {
                        logDeprecatedError(key, xKey, yKey);
                    }
                    this[xKey] = pos[0];
                    this[yKey] = pos[1];
                    this[privateKey] = pos;
                    enhanceArray(this, pos);
                }
            });
            function enhanceArray(self, pos) {
                Object.defineProperty(pos, 0, {
                    get: function () {
                        return self[xKey];
                    },
                    set: function (val) {
                        self[xKey] = val;
                    }
                });
                Object.defineProperty(pos, 1, {
                    get: function () {
                        return self[yKey];
                    },
                    set: function (val) {
                        self[yKey] = val;
                    }
                });
            }
        }
        if (Object.defineProperty) {
            createLegacyProperty('position', '_legacyPos', 'x', 'y');
            createLegacyProperty('scale', '_legacyScale', 'scaleX', 'scaleY');
            createLegacyProperty('origin', '_legacyOrigin', 'originX', 'originY');
        }
    })();
    return Element;
}());
mixin(Element, Eventful);
mixin(Element, Transformable);
function animateTo(animatable, target, cfg, animationProps, reverse) {
    cfg = cfg || {};
    var animators = [];
    animateToShallow(animatable, '', animatable, target, cfg, animationProps, animators, reverse);
    var finishCount = animators.length;
    var doneHappened = false;
    var cfgDone = cfg.done;
    var cfgAborted = cfg.aborted;
    var doneCb = function () {
        doneHappened = true;
        finishCount--;
        if (finishCount <= 0) {
            doneHappened
                ? (cfgDone && cfgDone())
                : (cfgAborted && cfgAborted());
        }
    };
    var abortedCb = function () {
        finishCount--;
        if (finishCount <= 0) {
            doneHappened
                ? (cfgDone && cfgDone())
                : (cfgAborted && cfgAborted());
        }
    };
    if (!finishCount) {
        cfgDone && cfgDone();
    }
    if (animators.length > 0 && cfg.during) {
        animators[0].during(function (target, percent) {
            cfg.during(percent);
        });
    }
    for (var i = 0; i < animators.length; i++) {
        var animator = animators[i];
        if (doneCb) {
            animator.done(doneCb);
        }
        if (abortedCb) {
            animator.aborted(abortedCb);
        }
        if (cfg.force) {
            animator.duration(cfg.duration);
        }
        animator.start(cfg.easing);
    }
    return animators;
}
function copyArrShallow(source, target, len) {
    for (var i = 0; i < len; i++) {
        source[i] = target[i];
    }
}
function is2DArray(value) {
    return isArrayLike(value[0]);
}
function copyValue(target, source, key) {
    if (isArrayLike(source[key])) {
        if (!isArrayLike(target[key])) {
            target[key] = [];
        }
        if (isTypedArray(source[key])) {
            var len = source[key].length;
            if (target[key].length !== len) {
                target[key] = new (source[key].constructor)(len);
                copyArrShallow(target[key], source[key], len);
            }
        }
        else {
            var sourceArr = source[key];
            var targetArr = target[key];
            var len0 = sourceArr.length;
            if (is2DArray(sourceArr)) {
                var len1 = sourceArr[0].length;
                for (var i = 0; i < len0; i++) {
                    if (!targetArr[i]) {
                        targetArr[i] = Array.prototype.slice.call(sourceArr[i]);
                    }
                    else {
                        copyArrShallow(targetArr[i], sourceArr[i], len1);
                    }
                }
            }
            else {
                copyArrShallow(targetArr, sourceArr, len0);
            }
            targetArr.length = sourceArr.length;
        }
    }
    else {
        target[key] = source[key];
    }
}
function isValueSame(val1, val2) {
    return val1 === val2
        || isArrayLike(val1) && isArrayLike(val2) && is1DArraySame(val1, val2);
}
function is1DArraySame(arr0, arr1) {
    var len = arr0.length;
    if (len !== arr1.length) {
        return false;
    }
    for (var i = 0; i < len; i++) {
        if (arr0[i] !== arr1[i]) {
            return false;
        }
    }
    return true;
}
function animateToShallow(animatable, topKey, animateObj, target, cfg, animationProps, animators, reverse) {
    var targetKeys = keys(target);
    var duration = cfg.duration;
    var delay = cfg.delay;
    var additive = cfg.additive;
    var setToFinal = cfg.setToFinal;
    var animateAll = !isObject(animationProps);
    var existsAnimators = animatable.animators;
    var animationKeys = [];
    for (var k = 0; k < targetKeys.length; k++) {
        var innerKey = targetKeys[k];
        var targetVal = target[innerKey];
        if (targetVal != null && animateObj[innerKey] != null
            && (animateAll || animationProps[innerKey])) {
            if (isObject(targetVal)
                && !isArrayLike(targetVal)
                && !isGradientObject(targetVal)) {
                if (topKey) {
                    if (!reverse) {
                        animateObj[innerKey] = targetVal;
                        animatable.updateDuringAnimation(topKey);
                    }
                    continue;
                }
                animateToShallow(animatable, innerKey, animateObj[innerKey], targetVal, cfg, animationProps && animationProps[innerKey], animators, reverse);
            }
            else {
                animationKeys.push(innerKey);
            }
        }
        else if (!reverse) {
            animateObj[innerKey] = targetVal;
            animatable.updateDuringAnimation(topKey);
            animationKeys.push(innerKey);
        }
    }
    var keyLen = animationKeys.length;
    if (!additive && keyLen) {
        for (var i = 0; i < existsAnimators.length; i++) {
            var animator = existsAnimators[i];
            if (animator.targetName === topKey) {
                var allAborted = animator.stopTracks(animationKeys);
                if (allAborted) {
                    var idx = indexOf(existsAnimators, animator);
                    existsAnimators.splice(idx, 1);
                }
            }
        }
    }
    if (!cfg.force) {
        animationKeys = filter(animationKeys, function (key) { return !isValueSame(target[key], animateObj[key]); });
        keyLen = animationKeys.length;
    }
    if (keyLen > 0
        || (cfg.force && !animators.length)) {
        var revertedSource = void 0;
        var reversedTarget = void 0;
        var sourceClone = void 0;
        if (reverse) {
            reversedTarget = {};
            if (setToFinal) {
                revertedSource = {};
            }
            for (var i = 0; i < keyLen; i++) {
                var innerKey = animationKeys[i];
                reversedTarget[innerKey] = animateObj[innerKey];
                if (setToFinal) {
                    revertedSource[innerKey] = target[innerKey];
                }
                else {
                    animateObj[innerKey] = target[innerKey];
                }
            }
        }
        else if (setToFinal) {
            sourceClone = {};
            for (var i = 0; i < keyLen; i++) {
                var innerKey = animationKeys[i];
                sourceClone[innerKey] = cloneValue(animateObj[innerKey]);
                copyValue(animateObj, target, innerKey);
            }
        }
        var animator = new Animator(animateObj, false, false, additive ? filter(existsAnimators, function (animator) { return animator.targetName === topKey; }) : null);
        animator.targetName = topKey;
        if (cfg.scope) {
            animator.scope = cfg.scope;
        }
        if (setToFinal && revertedSource) {
            animator.whenWithKeys(0, revertedSource, animationKeys);
        }
        if (sourceClone) {
            animator.whenWithKeys(0, sourceClone, animationKeys);
        }
        animator.whenWithKeys(duration == null ? 500 : duration, reverse ? reversedTarget : target, animationKeys).delay(delay || 0);
        animatable.addAnimator(animator, topKey);
        animators.push(animator);
    }
}
export default Element;
