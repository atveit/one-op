public class Tokenizer {
    
    /*@ public normal_behavior
      @ requires text != null;
      @ ensures \result != null;
      @ ensures \result.length == text.length;
      @ ensures (\forall int k; 0 <= k && k < text.length; \result[k] == (int)text[k]);
      @*/
    public int[] encode(char[] text) {
        int[] tokens = new int[text.length];
        
        /*@ loop_invariant 0 <= i && i <= text.length &&
          @ (\forall int j; 0 <= j && j < i; tokens[j] == (int)text[j]);
          @ assignable tokens[*];
          @ decreasing text.length - i;
          @*/
        for (int i = 0; i < text.length; i++) {
            tokens[i] = (int) text[i];
        }
        return tokens;
    }

    /*@ public normal_behavior
      @ requires tokens != null;
      @ ensures \result != null;
      @ ensures \result.length == tokens.length;
      @ ensures (\forall int k; 0 <= k && k < tokens.length; \result[k] == (char)tokens[k]);
      @*/
    public char[] decode(int[] tokens) {
        char[] chars = new char[tokens.length];
        
        /*@ loop_invariant 0 <= i && i <= tokens.length &&
          @ (\forall int j; 0 <= j && j < i; chars[j] == (char)tokens[j]);
          @ assignable chars[*];
          @ decreasing tokens.length - i;
          @*/
        for (int i = 0; i < tokens.length; i++) {
            chars[i] = (char) tokens[i];
        }
        return chars;
    }

    /*@ public normal_behavior
      @ requires text != null;
      @ ensures \result != null;
      @ ensures \result.length == text.length;
      @ ensures (\forall int k; 0 <= k && k < text.length; \result[k] == text[k]);
      @*/
    public char[] verifyInvertibility(char[] text) {
        return decode(encode(text));
    }
}
