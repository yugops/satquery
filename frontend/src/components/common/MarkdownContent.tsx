import React, { useMemo } from "react";
import { marked } from "marked";
import DOMPurify from "dompurify";
import { cn } from "@/lib/utils";

interface MarkdownContentProps {
  content: string;
  className?: string;
}

export const MarkdownContent: React.FC<MarkdownContentProps> = ({ content, className }) => {
  const html = useMemo(() => {
    if (!content) return "";
    try {
      // Configure marked for tables and breaks
      marked.setOptions({
        gfm: true,
        breaks: true,
      });
      const rawHtml = marked.parse(content) as string;
      return typeof window !== "undefined" ? DOMPurify.sanitize(rawHtml) : rawHtml;
    } catch {
      return content;
    }
  }, [content]);

  return (
    <div
      className={cn(
        "prose prose-invert max-w-none text-sm leading-relaxed text-zinc-200",
        // Headings
        "[&_h1]:text-xl [&_h1]:font-bold [&_h1]:text-white [&_h1]:mt-5 [&_h1]:mb-2.5",
        "[&_h2]:text-lg [&_h2]:font-semibold [&_h2]:text-white [&_h2]:mt-4 [&_h2]:mb-2",
        "[&_h3]:text-base [&_h3]:font-semibold [&_h3]:text-sky-400 [&_h3]:mt-4 [&_h3]:mb-1.5 [&_h3]:flex [&_h3]:items-center [&_h3]:gap-1.5",
        "[&_h4]:text-sm [&_h4]:font-semibold [&_h4]:text-zinc-300 [&_h4]:mt-3 [&_h4]:mb-1",
        // Paragraphs
        "[&_p]:mb-3 [&_p]:last:mb-0 [&_p]:leading-relaxed",
        // Lists
        "[&_ul]:my-2.5 [&_ul]:pl-5 [&_ul]:list-disc [&_ul]:space-y-1",
        "[&_ol]:my-2.5 [&_ol]:pl-5 [&_ol]:list-decimal [&_ol]:space-y-1",
        "[&_li]:text-zinc-300 [&_li]:leading-normal",
        // Tables
        "[&_table]:my-4 [&_table]:w-full [&_table]:border-collapse [&_table]:rounded-xl [&_table]:overflow-hidden [&_table]:border [&_table]:border-white/10 [&_table]:bg-white/[0.02]",
        "[&_th]:border [&_th]:border-white/10 [&_th]:bg-white/10 [&_th]:px-3 [&_th]:py-2 [&_th]:text-left [&_th]:text-xs [&_th]:font-semibold [&_th]:text-zinc-200",
        "[&_td]:border [&_td]:border-white/10 [&_td]:px-3 [&_td]:py-1.5 [&_td]:text-xs [&_td]:text-zinc-300 [&_td]:font-mono",
        // Strong & Emphasis
        "[&_strong]:font-semibold [&_strong]:text-white",
        "[&_em]:text-zinc-300 [&_em]:italic",
        // Code & Pre
        "[&_code]:rounded-md [&_code]:bg-white/10 [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:font-mono [&_code]:text-xs [&_code]:text-sky-300",
        "[&_pre]:my-3 [&_pre]:overflow-x-auto [&_pre]:rounded-xl [&_pre]:border [&_pre]:border-white/10 [&_pre]:bg-black/50 [&_pre]:p-3",
        // Blockquotes
        "[&_blockquote]:my-3 [&_blockquote]:border-l-2 [&_blockquote]:border-primary/50 [&_blockquote]:bg-primary/5 [&_blockquote]:pl-3 [&_blockquote]:py-1 [&_blockquote]:text-zinc-300 [&_blockquote]:italic",
        // Horizontal Rules
        "[&_hr]:my-4 [&_hr]:border-white/10",
        className,
      )}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
};

export default MarkdownContent;
