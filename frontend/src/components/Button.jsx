import React, { forwardRef } from 'react';
import { LoaderCircle } from 'lucide-react';

// One height per size, whatever the label, icon or state: a button that
// changes its text (Activate → Loading… → Active) never changes its box.
const SIZES = {
  sm: 'h-8 gap-1.5 px-3 text-xs',
  md: 'h-9 gap-2 px-3.5 text-xs',
  lg: 'h-11 gap-2 px-5 text-sm font-semibold',
};
const ICON_SIZES = { sm: 'h-8 w-8', md: 'h-9 w-9', lg: 'h-11 w-11' };
const GLYPH = { sm: 'h-3.5 w-3.5', md: 'h-4 w-4', lg: 'h-4 w-4' };
// Variants map onto the frost-control tones in index.css.
const TONES = { primary: 'primary', danger: 'danger', success: 'success', warning: 'warning' };

/**
 * The studio's only button.
 *
 * `icon` is a lucide component shown before the label; without children the
 * button is a square icon button (give it an aria-label). `loading` swaps the
 * icon for a spinner and disables the button without resizing it. Pass `href`
 * to render a link that looks the same.
 */
const Button = forwardRef(function Button({
  variant = 'secondary', size = 'sm', icon: Icon = null, loading = false, disabled = false, selected,
  className = '', children, href, type = 'button', ...rest
}, ref) {
  const iconOnly = children == null || children === false;
  const classes = [
    'frost-control inline-flex shrink-0 items-center justify-center whitespace-nowrap font-medium',
    iconOnly ? `${ICON_SIZES[size]} p-0` : SIZES[size],
    variant === 'ghost' ? 'ghost-control' : '',
    TONES[variant] ? '' : 'text-fg-muted',
    className,
  ].filter(Boolean).join(' ');
  const glyph = loading
    ? <LoaderCircle className={`${GLYPH[size]} shrink-0 animate-spin`} aria-hidden="true" />
    : Icon ? <Icon className={`${GLYPH[size]} shrink-0`} aria-hidden="true" /> : null;
  const content = (
    <>
      {glyph}
      {iconOnly ? null : <span className="min-w-0 truncate">{children}</span>}
    </>
  );
  const shared = {
    ref, className: classes, 'data-tone': TONES[variant], 'data-selected': selected, ...rest,
  };
  if (href) {
    return <a href={disabled ? undefined : href} aria-disabled={disabled || undefined} {...shared}>{content}</a>;
  }
  return (
    <button type={type} disabled={disabled || loading} aria-busy={loading || undefined} {...shared}>
      {content}
    </button>
  );
});

export default Button;
