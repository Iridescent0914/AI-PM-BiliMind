"use client";

import { useState, useEffect } from "react";
import { authApi, QRCodeResponse, UserInfo } from "@/lib/api";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: (sessionId: string, user: UserInfo) => void;
}

export default function LoginModal({ isOpen, onClose, onSuccess }: Props) {
  const [qr, setQr] = useState<QRCodeResponse | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "scanned" | "success" | "expired" | "error">("loading");
  const [polling, setPolling] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const getQR = async () => {
    setPolling(false);
    setQr(null);
    setErrorMessage("");
    setStatus("loading");
    try {
      const data = await authApi.getQRCode();
      setQr(data);
      setStatus("ready");
      setPolling(true);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "无法连接二维码服务");
      setStatus("error");
    }
  };

  useEffect(() => {
    if (isOpen) getQR();
    else {
      setPolling(false);
      setQr(null);
    }
  }, [isOpen]);

  useEffect(() => {
    if (!polling || !qr) return;
    let active = true;
    let inFlight = false;
    const timer = setInterval(async () => {
      if (inFlight) return;
      inFlight = true;
      try {
        const res = await authApi.pollQRCode(qr.qrcode_key);
        if (!active) return;
        if (res.status === "scanned") setStatus("scanned");
        else if (res.status === "confirmed") {
          setPolling(false);
          setStatus("success");
          setTimeout(() => onSuccess(res.session_id!, res.user_info!), 500);
        } else if (res.status === "expired") {
          setPolling(false);
          setStatus("expired");
        }
      } catch (error) {
        if (!active) return;
        setPolling(false);
        setErrorMessage(error instanceof Error ? error.message : "无法查询扫码状态");
        setStatus("error");
      } finally {
        inFlight = false;
      }
    }, 2000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [polling, qr, onSuccess]);

  if (!isOpen) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <h2 className="modal-title">扫码登录</h2>
        <p className="modal-subtitle">使用哔哩哔哩 APP 扫描</p>

        <div className="mt-4 flex justify-center">
          {status === "loading" && (
            <div className="w-48 h-48 flex items-center justify-center border border-dashed border-[var(--border)] rounded-2xl">
              <div className="w-8 h-8 border-2 border-[var(--accent)] border-t-transparent rounded-full animate-spin" />
            </div>
          )}

          {(status === "ready" || status === "scanned") && qr && (
            <div className="relative">
              <img src={qr.qrcode_image_base64} alt="二维码" className="w-48 h-48 rounded-2xl border border-[var(--border)]" />
              {status === "scanned" && (
                <div className="absolute inset-0 bg-white/90 rounded-2xl flex flex-col items-center justify-center">
                  <div className="status-pill">已扫码</div>
                  <span className="text-sm mt-3">请在手机上确认</span>
                </div>
              )}
            </div>
          )}

          {status === "success" && (
            <div className="w-48 h-48 flex flex-col items-center justify-center">
              <div className="status-pill">登录成功</div>
              <p className="text-sm text-[var(--muted)] mt-3">正在进入工作台</p>
            </div>
          )}

          {(status === "expired" || status === "error") && (
            <div className="w-48 h-48 flex flex-col items-center justify-center">
              <p className="text-sm text-[var(--muted)] mb-3 text-center">
                {status === "expired" ? "二维码已过期" : "二维码请求失败"}
              </p>
              {status === "error" && errorMessage && (
                <p className="text-xs text-[var(--muted)] mb-3 break-all text-center">{errorMessage}</p>
              )}
              <button onClick={getQR} className="btn btn-primary">重新获取</button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
