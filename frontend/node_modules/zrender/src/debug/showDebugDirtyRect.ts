import type { ZRenderType } from '../zrender';
import type CanvasPainter from '../canvas/Painter';
import type BoundingRect from '../core/BoundingRect';
import { extend } from '../core/util';

class DebugRect {

    dom: HTMLDivElement

    private _hideTimeout: number

    constructor(style: Opts['style']) {
        const dom = this.dom = document.createElement('div');
        dom.className = 'ec-debug-dirty-rect';

        style = extend({}, style);
        extend(style, {
            backgroundColor: 'rgba(0, 0, 255, 0.2)',
            border: '1px solid #00f'
        });
        dom.style.cssText = `
position: absolute;
opacity: 0;
transition: opacity 0.5s linear;
pointer-events: none;
`;

        for (let key in style) {
            if (style.hasOwnProperty(key)) {
                (dom.style as any)[key] = (style as any)[key];
            }
        }
    }

    update(rect: BoundingRect) {
        const domStyle = this.dom.style;
        domStyle.width = rect.width + 'px';
        domStyle.height = rect.height + 'px';
        domStyle.left = rect.x + 'px';
        domStyle.top = rect.y + 'px';
    }

    hide() {
        this.dom.style.opacity = '0';
    }
    show(autoHideDelay?: number) {
        clearTimeout(this._hideTimeout);

        this.dom.style.opacity = '1';

        // Auto hide after 2 second
        this._hideTimeout = setTimeout(() => {
            this.hide();
        }, autoHideDelay || 1000) as unknown as number;
    }

}

interface Opts {
    style?: {
        backgroundColor?: string
        color?: string
    }

    autoHideDelay?: number
}

export default function showDebugDirtyRect(zr: ZRenderType, opts?: Opts) {
    opts = opts || {};
    const painter = zr.painter as CanvasPainter;
    if (!painter.getLayers) {
        throw new Error('Debug dirty rect can only been used on canvas renderer.');
    }
    if (painter.isSingleCanvas()) {
        throw new Error('Debug dirty rect can only been used on zrender inited with container.');
    }
    const debugViewRoot = document.createElement('div');
    debugViewRoot.style.cssText = `
position:absolute;
left:0;
top:0;
right:0;
bottom:0;
pointer-events:none;
`;
    debugViewRoot.className = 'ec-debug-dirty-rect-container';

    const debugRects: DebugRect[] = [];
    const dom = zr.dom;
    dom.appendChild(debugViewRoot);
    const computedStyle = getComputedStyle(dom);
    if (computedStyle.position === 'static') {
        dom.style.position = 'relative';
    }

    zr.on('rendered', function () {
        if (painter.getLayers) {
            let idx = 0;
            painter.eachBuiltinLayer((layer) => {
                if (!layer.debugGetPaintRects) {
                    return;
                }
                const paintRects = layer.debugGetPaintRects();
                for (let i = 0; i < paintRects.length; i++) {
                    if (!paintRects[i].width || !paintRects[i].height) {
                        continue;
                    }

                    if (!debugRects[idx]) {
                        debugRects[idx] = new DebugRect(opts.style);
                        debugViewRoot.appendChild(debugRects[idx].dom);
                    }
                    debugRects[idx].show(opts.autoHideDelay);
                    debugRects[idx].update(paintRects[i]);
                    idx++;
                }
            });
            for (let i = idx; i < debugRects.length; i++) {
                debugRects[i].hide();
            }
        }
    });
}