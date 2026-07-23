import env from './core/env';

let dpr = 1;

// If in browser environment
if (env.hasGlobalWindow) {
    dpr = Math.max(
        window.devicePixelRatio
        || (window.screen && (window.screen as any).deviceXDPI / (window.screen as any).logicalXDPI)
        || 1, 1
    );
}

/**
 * Debug log mode:
 * 0: Do nothing, for release.
 * 1: console.error, for debug.
 */
export const debugMode = 0;

// retina 屏幕优化
export const devicePixelRatio = dpr;


/**
 * Determine when to turn on dark mode based on the luminance of backgroundColor
 */
export const DARK_MODE_THRESHOLD = 0.4;

/**
 * Color of default dark label.
 */
export const DARK_LABEL_COLOR = '#333';

/**
 * Color of default light label.
 */
export const LIGHT_LABEL_COLOR = '#ccc';

/**
 * Color of default light label.
 */
export const LIGHTER_LABEL_COLOR = '#eee';
