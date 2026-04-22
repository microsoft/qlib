import type { ZRenderType } from '../zrender';
interface Opts {
    style?: {
        backgroundColor?: string;
        color?: string;
    };
    autoHideDelay?: number;
}
export default function showDebugDirtyRect(zr: ZRenderType, opts?: Opts): void;
export {};
